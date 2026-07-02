"""Trainable bottlenecks on the per-modality adapter outputs of a frozen AV-LLM.

  ResidualBottleneck     (v0) — deterministic zero-init residual (identity at init).
  VariationalBottleneck  (v1) — per-modality VIB: z = mu + sigma*eps, zero-init output
                                (identity at init), exposes a KL rate term for the IB loss.

Both attach via forward hooks on `model.adapter_modules()`, LLM + encoders frozen.
Zero-init output => attaching doesn't change the model until trained. A `bypass` flag
makes a bottleneck a pass-through (identity) — used to get the DPO reference logprobs
(the frozen base) without holding a second model. See ib-rl-method-and-framing.md (§2).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finmax(t):
    """'<finite?>|<max|abs|>' for the VIB debug trace."""
    ok = bool(torch.isfinite(t).all())
    return f"{ok}|{(float(t.abs().max()) if ok else float('nan')):.1e}"


class ResidualBottleneck(nn.Module):
    """y = x + W2(GELU(W1 x)), W2 zero-initialized -> identity at init."""

    def __init__(self, dim: int = 2048, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.bypass = False
        self.last_kl = None

    def forward(self, x):
        if self.bypass:
            return x
        return x + self.fc2(self.act(self.fc1(x)))


class VariationalBottleneck(nn.Module):
    """Per-modality variational information bottleneck on an adapter output (T, dim).

    z = mu(x) + sigma(x)*eps ;  y = x + W_out(z)  with W_out zero-init (identity at init).
    `last_kl` = KL(N(mu, sigma^2) || N(0, I)) (mean over tokens+dims) for the IB loss term.
    """

    def __init__(self, dim: int = 2048, hidden: int | None = None, normalize_input: bool = False):
        super().__init__()
        hidden = hidden or dim
        self.enc = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.to_mu = nn.Linear(hidden, hidden)
        self.to_logvar = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # VideoLLaMA2's disable_torch_init() no-ops nn.Linear.reset_parameters process-wide,
        # so enc/to_mu/to_logvar can come out UNINITIALIZED (garbage ~1e34 -> inf KL, nan z,
        # NaN logits). Re-init any Linear whose weight/bias is non-finite or absurdly large
        # (proper Kaiming is ~1/sqrt(fan) << 10); a backbone with a working default init
        # (Qwen-Omni) leaves these well below threshold, so it is untouched.
        for lin in (self.enc, self.to_mu, self.to_logvar):
            w, b = lin.weight, lin.bias
            if (not torch.isfinite(w).all() or float(w.abs().max()) > 10.0
                    or not torch.isfinite(b).all() or float(b.abs().max()) > 10.0):
                nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)
                nn.init.zeros_(lin.bias)
        # Parameter-free LayerNorm on the ENCODER input only (the residual still carries the
        # raw x), so mu/logvar -> KL rate are scale-invariant. Needed for backbones with
        # "massive activations" (VideoLLaMA2, ~1e9 features); off by default so normal-scale
        # backbones (Qwen-Omni) are byte-for-byte unchanged. (See docs/research.)
        self.normalize_input = normalize_input
        self.bypass = False
        self.last_kl = None
        self.last_eps = None       # the reparam noise z used last forward (for fixed-eps replay)
        self._forced_eps = None    # if set (and shape-matching), reuse this noise instead of sampling
        self.last_kl_per_token = None  # (..., T) bits the bottleneck allocates per token
        self.last_residual_per_token = None    # (..., T) ||out(z)|| -> actual edit magnitude
        self.last_input_norm_per_token = None  # (..., T) ||x||      -> for the relative edit

    def forward(self, x):
        if self.bypass:
            return x
        pd = self.enc.weight.dtype          # fp32 on fp16 backbones -> no overflow / 0*inf NaN
        # Force the VIB to compute in its param dtype even under an outer bf16 autocast
        # (fp16 backbones): otherwise enc()/mu overflow bf16 and the KL rate blows up to inf.
        with torch.autocast(x.device.type, enabled=False):
            xc = torch.nan_to_num(x.to(pd))     # guard inf/nan backbone feats (enc() spreads NaN)
            enc_in = F.layer_norm(xc, (xc.shape[-1],)) if self.normalize_input else xc
            h = self.act(self.enc(enc_in))
            mu = self.to_mu(h)
            logvar = self.to_logvar(h).clamp(-8.0, 8.0)
            if self.training:
                # reuse a pinned eps when replaying a sample (fixed-eps GRPO -> the gradient
                # forward matches the sampling forward); else draw fresh reparam noise.
                if self._forced_eps is not None and self._forced_eps.shape == mu.shape:
                    eps = self._forced_eps.to(mu.dtype)
                else:
                    eps = torch.randn_like(mu)
                self.last_eps = eps.detach()
                z = mu + eps * torch.exp(0.5 * logvar)
            else:
                z = mu
            kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            self.last_kl = kl_elem.mean()                       # scalar rate for the IB loss
            if os.environ.get("RLVIB_VIB_DEBUG"):
                print(f"  [vibdbg] x_in={_finmax(x)} xc={_finmax(xc)} enc_in={_finmax(enc_in)} "
                      f"h={_finmax(h)} mu={_finmax(mu)} logvar={_finmax(logvar)} "
                      f"kl_elem={_finmax(kl_elem)} encW={_finmax(self.enc.weight)}", flush=True)
            self.last_kl_per_token = kl_elem.sum(dim=-1).detach()  # per-token rate -> saliency map
            delta = self.out(z)
            self.last_residual_per_token = delta.detach().norm(dim=-1)    # ||edit|| per token
            self.last_input_norm_per_token = xc.detach().norm(dim=-1)     # ||token|| (relative edit)
        return x + delta.to(x.dtype)


class FiLMVariationalBottleneck(VariationalBottleneck):
    """Prompt-aware VIB (scope-1 of docs/research/query-conditioned-bottleneck.md): FiLM-modulate
    the bottleneck hidden activation by a pooled question embedding q, plus a per-modality output
    gate g(q). Lets the module KEEP AUDIO when asked about sound and vision when asked about looks
    (selective, question-routed grounding) instead of the unconditional vision-rewrite.

        h          = GELU(enc(x))
        gamma,beta = film(q).chunk(2)               # film zero-init -> gamma=beta=0 at init
        h'         = (1 + gamma) * h + beta          # 1+gamma => identity at init; clamp gamma
        mu,logvar  = to_mu(h'), to_logvar(h')
        z          = mu + sigma*eps (train) | mu (eval)
        g          = sigmoid(gate(q))                # gate bias +4 => g~=0.98 (open) at init
        y          = x + g * out(z)                  # out zero-init => y == x at init, ANY q,g

    q is supplied out-of-band via set_condition() before each forward (the adapter hook only sees
    x, never the question). q is None -> behaves EXACTLY like VariationalBottleneck (unconditional
    / back-compat). Identity-at-init holds regardless of q, so bypass->exact-base and the
    anchored-DPO reference are untouched. Trained per-modality; both modalities share one q.
    """

    def __init__(self, dim: int = 2048, hidden: int | None = None, cond_dim: int = 2048,
                 normalize_input: bool = False):
        super().__init__(dim, hidden, normalize_input=normalize_input)
        hidden = hidden or dim
        self.cond_dim = cond_dim
        # FiLM generator q -> (gamma, beta). Zero-init => gamma=beta=0 at init, so the hidden
        # activation (hence mu/logvar/KL) is byte-for-byte the unconditional VIB at step 0;
        # gradients still flow (d h'/d gamma = h, d h'/d beta = 1) once `out` leaves zero.
        self.film = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # Output gate q -> 1 logit (scalar per modality, broadcast over tokens+features). Weight
        # zero + bias +4 => g = sigmoid(4) ~= 0.98 (open) and q-independent at init; `out` is zero
        # so the edit is 0 regardless of g, but starting "open" means the gate's job is to LEARN
        # TO CLOSE the irrelevant stream per question, descending from "both open" rather than
        # climbing the audio gate up from a dead 0 (a low-gradient flat region).
        self.gate = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 4.0)
        self._cond = None              # (cond_dim,) projected question vec; set by set_condition()
        self.last_gate = None          # live gate value -> optional gate-usage loss + routing probe

    def forward(self, x):
        if self.bypass:                # reference pass: exact frozen base (q ignored)
            return x
        q = self._cond
        pd = self.enc.weight.dtype     # fp32 on fp16/normalize backbones -> no overflow / dtype skew
        with torch.autocast(x.device.type, enabled=False):
            xc = torch.nan_to_num(x.to(pd))
            enc_in = F.layer_norm(xc, (xc.shape[-1],)) if self.normalize_input else xc
            h = self.act(self.enc(enc_in))
            if q is not None:
                qd = q.to(pd).reshape(-1)                       # (cond_dim,) -- batch-1 loop
                gamma, beta = self.film(qd).chunk(2, dim=-1)    # (hidden,), (hidden,)
                gamma = gamma.clamp(-1.0, 1.0)                  # scale in [0,2]: FiLM stability
                h = (1.0 + gamma) * h + beta                    # broadcast over all leading dims
            mu = self.to_mu(h)
            logvar = self.to_logvar(h).clamp(-8.0, 8.0)
            if self.training:                          # honor fixed-eps replay (GRPO) like the parent
                if self._forced_eps is not None and self._forced_eps.shape == mu.shape:
                    eps = self._forced_eps.to(mu.dtype)
                else:
                    eps = torch.randn_like(mu)
                self.last_eps = eps.detach()
                z = mu + eps * torch.exp(0.5 * logvar)
            else:
                z = mu
            kl_elem = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            self.last_kl = kl_elem.mean()
            self.last_kl_per_token = kl_elem.sum(dim=-1).detach()
            delta = self.out(z)
            if q is not None:
                g = torch.sigmoid(self.gate(qd))               # (1,) live scalar gate
                self.last_gate = g                             # live: gate-usage term reads it; probes detach
                delta = g * delta                              # uniform scalar gate over the edit
            self.last_residual_per_token = delta.detach().norm(dim=-1)
            self.last_input_norm_per_token = xc.detach().norm(dim=-1)
        return x + delta.to(x.dtype)


@torch.no_grad()
def question_embedding(model, prompt: str):
    """Mean-pooled FROZEN token embedding of the QUESTION TEXT ONLY (no media tokens) ->
    (embed_dim,) on model.device, fp32. Reuses the LLM's own input embedding table so q lives
    in the model's token space -- no new encoder, no new deps. Pass to set_condition()."""
    lm = getattr(model.model, "thinker", model.model)        # Qwen3 -> .thinker; else self
    tok = getattr(model, "tokenizer", None) or model.processor.tokenizer
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    emb = lm.get_input_embeddings()(ids)                     # (1, L, embed_dim); table is frozen
    return emb.float().mean(dim=1)[0]                        # (embed_dim,)


def set_condition(bottlenecks, q_emb) -> None:
    """Stash the per-example conditioning vector on every modality bottleneck before a forward.
    Projects the frozen question embedding through the shared `q_proj` once. No-op for plain
    (unconditional) bottlenecks, so it is safe to call unconditionally in train/eval loops.
    Call IDENTICALLY in training and eval, right before each model forward; the bypassed
    reference pass ignores the condition, so no toggling is needed around it."""
    if "q_proj" not in bottlenecks:                          # plain VIB / Residual: nothing to do
        return
    proj = bottlenecks["q_proj"]
    q = proj(q_emb.to(proj.weight.dtype))                    # (embed_dim,) -> (cond_dim,)
    for name, b in bottlenecks.items():
        if name != "q_proj":
            b._cond = q                                      # audio + vision share the same q


def attach_bottlenecks(model, dim: int | None = None, cls=ResidualBottleneck,
                       normalize_input: bool = False, cond_dim: int | None = None):
    """Freeze the model; attach trainable bottlenecks (`cls`) on the audio + vision
    adapters via forward hooks. `dim` defaults to `model.hidden_dim`. `normalize_input`
    LayerNorms the VIB encoder input (for massive-activation backbones, e.g. VideoLLaMA2).

    Returns (ModuleDict, handles). Detach with: for h in handles: h.remove().
    """
    dim = dim or getattr(model, "hidden_dim", 2048)
    for p in model.model.parameters():
        p.requires_grad_(False)

    # Compute the VIB in fp32 for fp16 backbones AND massive-/large-activation backbones
    # (normalize_input, e.g. VideoLLaMA2 run in bf16): bf16's 8-bit mantissa makes the KL
    # (mu^2 / exp(logvar)) and z blow up to inf even on normalized O(1) inputs -> 0*inf = NaN.
    # Qwen-Omni (bf16, normalize_input=False) stays bf16 and is byte-for-byte unchanged.
    vib_dtype = (torch.float32
                 if (getattr(model, "dtype", None) == torch.float16 or normalize_input)
                 else model.dtype)
    # normalize_input applies to every VIB subclass (incl. FiLM); cond_dim only to the FiLM one.
    kw = {"normalize_input": normalize_input} if issubclass(cls, VariationalBottleneck) else {}
    is_film = issubclass(cls, FiLMVariationalBottleneck)
    if is_film:
        lm = getattr(model.model, "thinker", model.model)          # frozen embed table -> cond_dim
        embed_dim = lm.get_input_embeddings().weight.shape[1]
        cond_dim = cond_dim or embed_dim
        kw["cond_dim"] = cond_dim
    bottlenecks = nn.ModuleDict({"audio": cls(dim, **kw), "vision": cls(dim, **kw)})
    if is_film:                                                     # shared q-projection (q is the
        bottlenecks["q_proj"] = nn.Linear(embed_dim, cond_dim)     # same for both modalities)
    bottlenecks = bottlenecks.to(model.device, vib_dtype)

    handles = []
    for name, adapter in model.adapter_modules().items():
        bn = bottlenecks[name]

        def hook(_module, _inputs, output, bn=bn):
            if isinstance(output, tuple):
                return (bn(output[0]),) + tuple(output[1:])
            return bn(output)

        handles.append(adapter.register_forward_hook(hook))
    return bottlenecks, handles


def total_kl(bottlenecks):
    """Sum of the bottlenecks' last KL rate terms (0.0 if none recorded)."""
    kls = [b.last_kl for b in bottlenecks.values() if getattr(b, "last_kl", None) is not None]
    return sum(kls) if kls else 0.0


def set_bypass(bottlenecks, on: bool) -> None:
    """Toggle pass-through (identity) on all bottlenecks — used for the DPO reference."""
    for name, b in bottlenecks.items():
        if name != "q_proj":                 # q_proj is a bare Linear, not a bottleneck
            b.bypass = on


def capture_eps(bottlenecks):
    """Snapshot the reparam noise each bottleneck used in its last forward (fixed-eps GRPO)."""
    return {k: b.last_eps for k, b in bottlenecks.items() if getattr(b, "last_eps", None) is not None}


def set_forced_eps(bottlenecks, eps_map) -> None:
    """Pin each bottleneck's next-forward reparam noise to `eps_map[name]` (None per key = resample).

    Pass None to clear all (free-running sampling). Used to replay a sampled GRPO action's noise so
    the gradient forward reproduces the sampling forward without holding every group graph at once.
    """
    for k, b in bottlenecks.items():
        b._forced_eps = (eps_map or {}).get(k)


def load_attached(model, ckpt_path):
    """Attach a TRAINED bottleneck checkpoint to `model` for inference (eval mode).

    Reads the saved class + dim, attaches via forward hooks (freezing the model), loads
    the weights, switches to eval (deterministic z=mu). Returns (bottlenecks, handles) —
    keep the reference alive for the duration of inference.
    """
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("cls") == "LoRAAdapter":                 # LoRA baseline: its own attach path
        from rlvib.models.lora import load_attached_lora
        return load_attached_lora(model, ck)
    cls = {"VariationalBottleneck": VariationalBottleneck,
           "ResidualBottleneck": ResidualBottleneck,
           "FiLMVariationalBottleneck": FiLMVariationalBottleneck}.get(
               ck.get("cls"), VariationalBottleneck)
    bottlenecks, handles = attach_bottlenecks(model, dim=ck.get("dim"), cls=cls,
                                              normalize_input=ck.get("normalize_input", False),
                                              cond_dim=ck.get("cond_dim"))
    bottlenecks.load_state_dict(ck["state_dict"])
    bottlenecks.eval()
    return bottlenecks, handles
