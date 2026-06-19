"""Model wrappers + factory.

Each wrapper implements the same interface: `message`, `generate`,
`adapter_modules`, `device`, `dtype`, `model`, `hidden_dim`. Imports are lazy so an
env with only one model's deps (e.g. the rlvib_vl2 env for VideoLLaMA2, which pins
transformers 4.42) doesn't choke importing the others.
"""
import importlib

_MODELS = {
    "qwen3-omni": ("rlvib.models.qwen3_omni", "QwenOmni"),
    "qwen2.5-omni": ("rlvib.models.qwen25_omni", "Qwen25Omni"),
    "videollama2": ("rlvib.models.videollama2", "VideoLLaMA2"),
    # Closed-API frontier baselines (benchmark-only; no VIB -- see rlvib.models.api_models).
    "gemini": ("rlvib.models.api_models", "GeminiModel"),
    "gpt4o": ("rlvib.models.api_models", "OpenAIModel"),
}
_ALIASES = {
    "qwen3": "qwen3-omni", "qwen3omni": "qwen3-omni",
    "qwen25-omni": "qwen2.5-omni", "qwen2.5": "qwen2.5-omni", "qwen25omni": "qwen2.5-omni",
    "vl2": "videollama2", "videollama2.1-7b-av": "videollama2",
    "gpt-4o": "gpt4o", "openai": "gpt4o", "google": "gemini", "gemini-pro": "gemini",
}

MODEL_NAMES = sorted(_MODELS)


def get_model(name: str = "qwen3-omni", **kwargs):
    """Instantiate a base model wrapper by name (see MODEL_NAMES / _ALIASES)."""
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in _MODELS:
        raise ValueError(f"unknown model '{name}'; choices: {MODEL_NAMES}")
    mod, cls = _MODELS[key]
    return getattr(importlib.import_module(mod), cls)(**kwargs)


__all__ = ["get_model", "MODEL_NAMES"]
