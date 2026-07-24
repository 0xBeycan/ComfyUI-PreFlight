"""
Qwen-VL model management for PreFlight: path resolution, auto-download, load,
cache, quantization, and deterministic generation.

Architecture inherited from ComfyUI-QwenVL (GPL-3.0) but reimplemented
independently against ``transformers`` / ``huggingface_hub`` — see README's
licensing note. The behaviour, not the source, is inherited.

IMPORTANT: every heavy dependency (``torch``, ``transformers``,
``huggingface_hub``, ``folder_paths``, ``comfy``) is imported lazily INSIDE the
functions that need it. The module top imports only stdlib, so ``import
model_manager`` — and therefore importing ``nodes`` during pytest collection —
never requires torch or a ComfyUI runtime. Only the two functions that actually
touch a GPU (``load_model``, ``run_observation``) pull those in.
"""

import gc
import json
import os
from pathlib import Path

_MODELS_JSON = os.path.join(os.path.dirname(__file__), "models.json")

# Combo option lists surfaced to the node UI.
QUANTIZATION_OPTIONS = ["4-bit", "8-bit", "None (FP16)"]
DEFAULT_QUANTIZATION = "8-bit"
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention_2"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]

# Module-level cache. Keyed by (name, quantization, attention_mode, device); a
# key change clears VRAM before the new load. Held here (module scope) so it
# survives across node executions when keep_model_loaded is on.
_CACHE = {"key": None, "model": None, "processor": None}


# ---------------------------------------------------------------------------
# Config (pure — no ComfyUI/torch; safe to unit-test)
# ---------------------------------------------------------------------------

def load_models_config():
    """Return ``(models_dict, ordered_names)``; ``_``-prefixed metadata keys are
    filtered out and the default model is placed first."""
    with open(_MODELS_JSON, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    models = {k: v for k, v in raw.items() if not k.startswith("_")}
    names = list(models)
    defaults = [n for n in names if models[n].get("default")]
    if defaults:
        d = defaults[0]
        names.remove(d)
        names.insert(0, d)
    return models, names


def model_names():
    """Ordered model names for the combo (default first)."""
    return load_models_config()[1]


# ---------------------------------------------------------------------------
# Path resolution (§4: reuse the shared dir; handle LLM / llm casing)
# ---------------------------------------------------------------------------

def resolve_qwen_vl_dir():
    """The ``.../models/LLM/Qwen-VL`` directory, resolved defensively.

    Order: (1) a folder registered under "LLM"/"llm" in ComfyUI's
    folder_names_and_paths; (2) an already-existing ``models_dir/LLM`` or
    ``models_dir/llm``; (3) the canonical ``models_dir/LLM``. Checking both
    cases means a model another Qwen-VL pack put under ``llm`` is still found
    and reused instead of re-downloaded.
    """
    import folder_paths

    names_map = getattr(folder_paths, "folder_names_and_paths", {}) or {}
    for key in ("LLM", "llm"):
        if key in names_map:
            paths = folder_paths.get_folder_paths(key)
            if paths:
                return Path(paths[0]) / "Qwen-VL"

    models_dir = Path(folder_paths.models_dir)
    for key in ("LLM", "llm"):
        if (models_dir / key).exists():
            return models_dir / key / "Qwen-VL"
    return models_dir / "LLM" / "Qwen-VL"


def model_dir_for(name):
    return resolve_qwen_vl_dir() / name


def is_downloaded(path):
    """A model counts as present if it has a config.json AND at least one weight
    shard — so a half-finished download does not masquerade as complete."""
    path = Path(path)
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def ensure_downloaded(name, cfg):
    """Download the model into the shared dir if absent; return its path.

    No-op when the model is already present (acceptance §9.2) — the shared
    ``models/LLM/Qwen-VL`` location is exactly what makes that reuse possible.
    """
    target = model_dir_for(name)
    if is_downloaded(target):
        return target

    from huggingface_hub import snapshot_download

    pbar = None
    try:
        from comfy.utils import ProgressBar
        pbar = ProgressBar(100)
    except Exception:
        pbar = None  # running outside ComfyUI; HF prints its own console progress

    print("[PreFlight] downloading %s (%s) -> %s" % (name, cfg.get("repo_id"), target))
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg["repo_id"],
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    if pbar is not None:
        pbar.update_absolute(100, 100)
    return target


# ---------------------------------------------------------------------------
# Load + cache
# ---------------------------------------------------------------------------

def _resolve_device(device):
    import torch

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[PreFlight] cuda requested but unavailable — using cpu")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clear_cache():
    """Move the cached model off the GPU and drop it, then reclaim VRAM."""
    model = _CACHE.get("model")
    if model is not None:
        try:
            _CACHE["model"] = model.cpu()
        except Exception:
            pass
    _CACHE["model"] = None
    _CACHE["processor"] = None
    _CACHE["key"] = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _build_quant_config(quantization):
    """BitsAndBytesConfig for the chosen precision, or None for FP16/FP32."""
    import torch
    from transformers import BitsAndBytesConfig

    if quantization == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def _resolve_attention(attention_mode, force_sdpa):
    """FP8 / bitsandbytes / cpu paths must force sdpa (§2)."""
    if force_sdpa:
        return "sdpa"
    if attention_mode == "flash_attention_2":
        return "flash_attention_2"
    if attention_mode == "auto":
        return "flash_attention_2"  # optimistic; load falls back to sdpa on failure
    return "sdpa"


def load_model(name, quantization, attention_mode, device, keep_model_loaded):
    """Load (or reuse) a Qwen-VL model + processor. Returns ``(model, processor)``.

    Reuses the module cache when (name, quantization, attention_mode, device)
    is unchanged and keep_model_loaded is on; otherwise clears the old model
    from VRAM first.
    """
    resolved_device = _resolve_device(device)
    key = (name, quantization, attention_mode, resolved_device)
    if keep_model_loaded and _CACHE["key"] == key and _CACHE["model"] is not None:
        return _CACHE["model"], _CACHE["processor"]

    _clear_cache()  # key changed (or reload requested) — free the old model

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _VLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as _VLM

    models, _ = load_models_config()
    cfg = models[name]
    target = ensure_downloaded(name, cfg)

    quant_config = _build_quant_config(quantization)
    force_sdpa = (quant_config is not None
                  or cfg.get("quantized", False)
                  or resolved_device == "cpu")
    attn = _resolve_attention(attention_mode, force_sdpa)

    load_kwargs = {"trust_remote_code": True, "attn_implementation": attn}
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        if resolved_device == "cuda":
            load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = (torch.float16 if resolved_device == "cuda"
                                      else torch.float32)

    model = _from_pretrained(_VLM, str(target), load_kwargs)
    if quant_config is None:
        model = model.to(resolved_device)
    model.eval()

    processor = AutoProcessor.from_pretrained(str(target), trust_remote_code=True)

    _CACHE.update(key=key, model=model, processor=processor)
    return model, processor


def _from_pretrained(_VLM, path, load_kwargs):
    """from_pretrained with a graceful flash_attention_2 -> sdpa fallback."""
    try:
        return _VLM.from_pretrained(path, **load_kwargs)
    except Exception as exc:  # flash-attn missing, incompatible, etc.
        if load_kwargs.get("attn_implementation") == "flash_attention_2":
            print("[PreFlight] flash_attention_2 unavailable, falling back to sdpa "
                  "(%s)" % exc)
            fallback = dict(load_kwargs, attn_implementation="sdpa")
            return _VLM.from_pretrained(path, **fallback)
        raise


def release(keep_model_loaded):
    """Called after a run: if the user does not want the model kept resident,
    free it now so VRAM is returned between queue prompts."""
    if not keep_model_loaded:
        _clear_cache()


# ---------------------------------------------------------------------------
# Generation (deterministic / greedy)
# ---------------------------------------------------------------------------

def run_observation(model, processor, pil_frames, prompt, max_new_tokens=300,
                    seed=0, extra_instruction=""):
    """Run the sensor on one multi-image message and return the raw text.

    Greedy decoding (do_sample=False, num_beams=1, no temperature/top_p) with a
    fixed seed: this is a classification task, and sampling would inject run-to-
    run drift that makes calibration impossible (§4). All sampled frames go in a
    SINGLE model call so the model can infer motion across them.
    """
    import torch

    user_text = prompt
    if extra_instruction:
        user_text = prompt + "\n\n" + extra_instruction

    content = [{"type": "image", "image": img} for img in pil_frames]
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]

    # Prefer the modern combined path (processes embedded images itself); fall
    # back to the two-step template+processor path on older transformers.
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    except (TypeError, ValueError):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_frames, return_tensors="pt")

    inputs = inputs.to(model.device)

    torch.manual_seed(seed)
    with torch.no_grad():
        generated = model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)

    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated[:, prompt_len:]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip() if decoded else ""
