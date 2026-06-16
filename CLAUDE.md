# RLVIB

## Environment

Create / update the conda env from `environment.yml`:

```bash
mamba env create -f environment.yml             # first time
mamba env update -f environment.yml --prune     # after editing the file
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
