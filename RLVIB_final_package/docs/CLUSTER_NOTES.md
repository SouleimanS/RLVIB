# RLVIB

Clean-room rebuild of an earlier audio-visual grounding project (`av_ib`): the
prior code and the local AVQA dataset copy are corrupted and are **not** reused.
Everything here is implemented from scratch against a freshly sourced dataset.

## Environment

Create / update the conda env from `environment.yml`:

```bash
conda env create -f environment.yml             # first time
conda env update -f environment.yml --prune     # after editing the file
```

### Activate

**ABCI cluster** (account `aab11336im`) — the shell isn't pre-configured for
`conda activate`, so source conda's profile script first, then activate:

```bash
source /home/aab11336im/anaconda3/etc/profile.d/conda.sh
conda activate rlvib
```

On a machine where `conda activate` already works in your shell, just:

```bash
conda activate rlvib
```

## Running on ABCI (PBS / qsub)

GPUs are only available inside a batch/interactive job on the `rt_HF` queue —
login nodes are CPU-only (so `torch.cuda.is_available()` is `False` there, which
is expected). Job scripts live in `scripts/`.

**Verified working (2026-06):** `torch 2.12.0+cu130`, `cuda: True` on
8× NVIDIA H200 (~141 GB each), ABCI driver `580.105.08` (CUDA 13.0). The cu130
wheel runs fine on this driver — no cu12 build needed.

```bash
qsub scripts/gpucheck.qsub                                      # ~5-min GPU sanity check
qsub -I -P gae50891 -q rt_HF -l select=1 -l walltime=01:00:00   # interactive GPU session
qsub scripts/rlvib.qsub                                         # submit a batch job
qstat -u "$USER"                                                # check queue status
```

Job-script conventions (see `scripts/rlvib.qsub`): `cd "$PBS_O_WORKDIR"`, source
conda + `conda activate rlvib`, set `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:…"` so the
pip torch wheel finds its bundled CUDA libs, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
and `tee` output into `runs/`. Group code `-P gae50891`, queue `-q rt_HF`.

### Gotcha: don't export `LD_LIBRARY_PATH` shell-wide

Exporting `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:…"` in an interactive shell puts
conda's OpenSSL ahead of the system one and breaks system `git` over HTTPS:

```
git-remote-https: symbol lookup error: /lib64/libldap.so.2:
undefined symbol: EVP_md2, version OPENSSL_3.0.0
```

Run git with it cleared — `env -u LD_LIBRARY_PATH git …` — or just don't export
it interactively (torch's pip wheel finds its CUDA libs via RPATH, so it isn't
needed to run torch). It stays correctly scoped inside the qsub jobs.
