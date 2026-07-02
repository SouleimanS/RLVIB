"""LoRA baseline on the SAME attach points as the VIB bottleneck (the reviewer control).

Answers "why a VIB adapter and not LoRA?" with a like-for-like comparison: wrap every
nn.Linear inside the per-modality adapters (model.adapter_modules(): audio_tower.proj2 /
visual.merger) with a zero-init low-rank delta

    y = W x + (alpha/r) * B(A(x)),      B zero-init  =>  identity at init,

attached via forward hooks (no module surgery, same idiom as attach_bottlenecks), backbone
frozen. Each wrapper has a `bypass` flag, so `set_bypass` and the anchored-DPO reference
(= exact frozen base) work unchanged; `last_kl` is None, so `total_kl` contributes 0 (no IB
rate term -- that is the point of the control). Train with the identical anchored swap-DPO
recipe via `train_swap_anchored.py --lora`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Low-rank residual for one wrapped Linear: delta(x) = (alpha/r) * B(A(x))."""

    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)                    # zero-init => identity at init
        self.scale = alpha / r
        self.r, self.alpha = r, alpha
        self.bypass = False
        self.last_kl = None                              # total_kl() -> 0 for LoRA (no rate term)

    def forward(self, x):
        pd = self.A.weight.dtype
        with torch.autocast(x.device.type, enabled=False):   # match the VIB's dtype island
            delta = self.scale * self.B(self.A(torch.nan_to_num(x.to(pd))))
        return delta.to(x.dtype)


def attach_lora(model, r: int = 16, alpha: float = 32.0):
    """Freeze the model; wrap every nn.Linear under each per-modality adapter with a
    zero-init LoRA delta via forward hooks. Returns (ModuleDict, handles) -- the same
    contract as attach_bottlenecks, so set_bypass / the trainer / eval all work as-is."""
    for p in model.model.parameters():
        p.requires_grad_(False)
    lora_dtype = torch.float32 if getattr(model, "dtype", None) == torch.float16 else model.dtype

    adapters = nn.ModuleDict()
    handles = []
    for mname, mod in model.adapter_modules().items():
        for sub, lin in mod.named_modules():             # '' = the module itself if it IS a Linear
            if not isinstance(lin, nn.Linear):
                continue
            key = f"{mname}_{sub.replace('.', '_')}".rstrip("_")
            ad = LoRAAdapter(lin.in_features, lin.out_features, r=r, alpha=alpha)
            adapters[key] = ad

            def hook(_module, inputs, output, ad=ad):
                if ad.bypass:
                    return output
                return output + ad(inputs[0])

            handles.append(lin.register_forward_hook(hook))
    adapters = adapters.to(model.device, lora_dtype)
    return adapters, handles


def load_attached_lora(model, ck):
    """Attach + load a trained LoRA checkpoint for inference (mirror of load_attached)."""
    adapters, handles = attach_lora(model, r=ck.get("rank", 16), alpha=ck.get("alpha", 32.0))
    adapters.load_state_dict(ck["state_dict"])
    adapters.eval()
    return adapters, handles
