"""
ComfyUI adapter for PreFlight — the three nodes.

Thin by design: this file converts ComfyUI types (IMAGE tensors, widget values)
to/from plain data and orchestrates, but contains NO platform rules and NO store
format. All judgement lives in ``rules`` and all persistence in ``feedback``
(both pure stdlib). ``model_manager`` handles the GPU.

Import discipline (matches the sibling packs): the top of this file imports only
stdlib plus the pure/lazy PreFlight modules. ``torch`` is never imported (image
tensors are handled by duck typing); ``numpy``, ``PIL`` and ``folder_paths`` are
imported lazily inside the methods that need them, so ``import nodes`` — and
therefore pytest collecting the package — works with neither torch nor a ComfyUI
runtime installed.
"""

import itertools
import json
import traceback
from pathlib import Path

try:  # normal ComfyUI package import
    from . import rules, feedback, prompts, model_manager
except ImportError:  # imported without package context (pytest / tooling)
    import rules
    import feedback
    import prompts
    import model_manager

CATEGORY = "PreFlight"

# Fixed decode seed. Generation is greedy so this rarely matters, but pinning it
# removes one more source of run-to-run variation (§4).
_SEED = 42
_MAX_NEW_TOKENS = 300
_RETRY_INSTRUCTION = ("Your previous response was not valid JSON. Return ONLY "
                      "the JSON object, nothing else.")


# ---------------------------------------------------------------------------
# Pure helpers (testable without ComfyUI / torch)
# ---------------------------------------------------------------------------

def _extract_json(raw):
    """Return the first balanced ``{...}`` object parsed from ``raw``, or None.

    Scans for the first '{' and its matching '}', ignoring braces inside JSON
    strings — this transparently skips markdown fences and any prose the model
    wraps around the object.
    """
    if not isinstance(raw, str):
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except ValueError:
                    return None
    return None


def _meta(model_name, frames_analyzed):
    """The meta block the node injects into every valid observation."""
    return {
        "schema_version": prompts.SCHEMA_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "model_name": model_name,
        "frames_analyzed": int(frames_analyzed),
    }


def _observe_from_generate(generate, model_name, frames_analyzed):
    """Drive generation -> parse -> validate, with exactly one retry (§4).

    ``generate(extra_instruction)`` returns raw model text. On the first
    unparseable / non-observation response we retry once with the corrective
    instruction; a second failure returns ``{"error": ...}`` (no meta — the
    error key alone makes rules.judge fail closed). Returns ``(observations,
    raw_response)``.
    """
    raw = generate("")
    parsed = _extract_json(raw)

    if not prompts.has_observation_keys(parsed):
        raw2 = generate(_RETRY_INSTRUCTION)
        raw = raw + "\n\n--- retry ---\n\n" + raw2
        parsed = _extract_json(raw2)
        if not prompts.has_observation_keys(parsed):
            return {"error": "model did not return valid observations JSON after "
                             "one retry"}, raw

    observations = prompts.validate_observations(parsed)
    observations["meta"] = _meta(model_name, frames_analyzed)
    return observations, raw


def feedback_store_path():
    """ComfyUI/output/preflight/feedback.jsonl, or the standalone default when
    folder_paths is unavailable (outside ComfyUI). Kept as a seam tests patch."""
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory()) / "preflight" / "feedback.jsonl"
    except Exception:
        return feedback.DEFAULT_STORE


def _safe_model_names():
    try:
        names = model_manager.model_names()
        return names or ["Qwen3-VL-8B-Instruct"]
    except Exception:
        return ["Qwen3-VL-8B-Instruct"]


# ---------------------------------------------------------------------------
# Tensor -> PIL (duck-typed: works for torch tensors and numpy arrays)
# ---------------------------------------------------------------------------

def _to_numpy(frame):
    import numpy as np
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    return np.asarray(frame)


def _frame_to_pil(frame):
    import numpy as np
    from PIL import Image
    arr = np.clip(_to_numpy(frame) * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    channels = arr.shape[2]
    if channels == 1:
        return Image.fromarray(arr[:, :, 0], "L").convert("RGB")
    if channels == 4:
        return Image.fromarray(arr, "RGBA").convert("RGB")
    return Image.fromarray(arr[:, :, :3], "RGB")


def _sample_frames(image, max_frames):
    """Evenly sample up to ``max_frames`` frames from an IMAGE batch -> PIL list.

    A single call later sends them all together so the model can infer motion.
    """
    import numpy as np
    batch = int(image.shape[0])
    count = max(1, min(int(max_frames), batch))
    idxs = np.unique(np.linspace(0, batch - 1, count).round().astype(int))
    return [_frame_to_pil(image[i]) for i in idxs]


# ---------------------------------------------------------------------------
# Node 1 — Observe
# ---------------------------------------------------------------------------

class PreFlightObserve:
    CATEGORY = CATEGORY
    FUNCTION = "observe"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("observations_json", "raw_response")
    DESCRIPTION = (
        "Run a local Qwen-VL model as a pure observation sensor. It describes "
        "what is visually present (garment, exposure, framing, pose, in-image "
        "text, ...) as a strict JSON object — it does NOT judge acceptability. "
        "Feed the result into PreFlight: Report.\n\n"
        "Deterministic: greedy decoding, so the same image + settings give "
        "byte-identical output run to run. Errors never break the graph — a "
        "problem comes back as {\"error\": ...} so Report can fail closed.")

    @classmethod
    def INPUT_TYPES(cls):
        names = _safe_model_names()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A single image or a video frame "
                                               "batch."}),
                "model_name": (names, {"default": names[0],
                    "tooltip": "Qwen-VL model (from models.json). Downloaded once "
                               "into models/LLM/Qwen-VL and reused across packs."}),
                "quantization": (model_manager.QUANTIZATION_OPTIONS, {
                    "default": model_manager.DEFAULT_QUANTIZATION,
                    "tooltip": "4-bit / 8-bit (bitsandbytes) shrink VRAM; None keeps "
                               "FP16. Quantized paths force sdpa attention."}),
                "attention_mode": (model_manager.ATTENTION_OPTIONS, {
                    "default": "auto",
                    "tooltip": "auto tries flash_attention_2 then falls back to sdpa. "
                               "Quantized / CPU always use sdpa."}),
                "max_frames": ("INT", {"default": 6, "min": 1, "max": 16,
                    "tooltip": "For video: how many frames to sample evenly from the "
                               "batch. All are sent in one model call."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the model resident in VRAM between runs. Turn "
                               "off to free VRAM after each run."}),
                "device": (model_manager.DEVICE_OPTIONS, {"default": "auto",
                    "tooltip": "auto picks cuda when available, else cpu."}),
            }
        }

    def observe(self, image, model_name, quantization, attention_mode, max_frames,
                keep_model_loaded, device):
        raw = ""
        try:
            frames = _sample_frames(image, max_frames)
            model, processor = model_manager.load_model(
                model_name, quantization, attention_mode, device, keep_model_loaded)

            def generate(extra_instruction):
                return model_manager.run_observation(
                    model, processor, frames, prompts.build_prompt(),
                    max_new_tokens=_MAX_NEW_TOKENS, seed=_SEED,
                    extra_instruction=extra_instruction)

            observations, raw = _observe_from_generate(generate, model_name, len(frames))
        except Exception as exc:  # errors never raise into the graph (§4)
            traceback.print_exc()
            observations = {"error": "%s: %s" % (type(exc).__name__, exc)}
        finally:
            try:
                model_manager.release(keep_model_loaded)
            except Exception:
                pass

        return (json.dumps(observations, ensure_ascii=False), raw)


# ---------------------------------------------------------------------------
# Node 2 — Report
# ---------------------------------------------------------------------------

class PreFlightReport:
    CATEGORY = CATEGORY
    FUNCTION = "report"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("report_json", "summary", "record_id", "image")
    DESCRIPTION = (
        "Apply the PreFlight rules engine to a sensor observation and predict, "
        "per platform (Instagram / TikTok / X), whether content risks removal or "
        "reach suppression. Outputs a full JSON report, a human-readable summary, "
        "and a feedback record id.\n\n"
        "This is a prediction, not an approval gate. Verdicts are ranges "
        "(best..worst); range_drivers name the unknown that would collapse them.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observations_json": ("STRING", {"forceInput": True,
                    "tooltip": "The observations_json output of PreFlight: Observe."}),
                "caption": ("STRING", {"default": "", "multiline": True,
                    "tooltip": "Optional caption to be published. Scanned for adult "
                               "solicitation / links together with in-image text."}),
                "log_prediction": ("BOOLEAN", {"default": True,
                    "tooltip": "Append this prediction to the feedback store so you "
                               "can later record what the platform actually did."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional pass-through so this node "
                                               "can sit inline in a workflow."}),
            },
        }

    def report(self, observations_json, caption="", log_prediction=True, image=None):
        try:
            observations = json.loads(observations_json)
        except (ValueError, TypeError):
            observations = {"error": "observations_json was not valid JSON"}
        if not isinstance(observations, dict):
            observations = {"error": "observations_json was not a JSON object"}

        meta = observations.get("meta") if isinstance(observations, dict) else None
        frames = meta.get("frames_analyzed", 1) if isinstance(meta, dict) else 1
        is_video = isinstance(frames, int) and frames > 1

        report = rules.judge(observations, caption or "", is_video=is_video)

        record_id = ""
        if log_prediction:
            try:
                record_id = feedback.log_prediction(
                    report, caption or "", path=feedback_store_path())
                report["record_id"] = record_id  # inject before summary (§7.3)
            except Exception as exc:  # store failure must not break the graph (§9.12)
                print("[PreFlight] feedback store write failed: %s" % exc)
                record_id = ""

        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        summary = rules.summary_text(report)
        return (report_json, summary, record_id, image)


# ---------------------------------------------------------------------------
# Node 3 — Outcome (feedback loop)
# ---------------------------------------------------------------------------

def _format_record_label(rec):
    """One combo line per prediction, built entirely from the record itself so
    no filename convention is needed to find it later (§7.4)."""
    rid = rec.get("id", "????????")
    ts = rec.get("ts", "")
    when = ts[5:16].replace("T", " ") if len(ts) >= 16 else ts  # MM-DD HH:MM
    verdicts = rec.get("verdicts", {})
    worst = " ".join("%s:%s" % (short, verdicts.get(p, {}).get("worst", "?"))
                     for p, short in (("instagram", "IG"), ("tiktok", "TT"), ("x", "X")))
    obs = rec.get("observations", {}) or {}
    what = "%s/%s" % (obs.get("garment", "?"), obs.get("setting", "?"))
    excerpt = (rec.get("caption_excerpt", "") or "").replace("\n", " ")
    if len(excerpt) > 18:
        excerpt = excerpt[:17] + "…"
    return "%s · %s · %s · %s · \"%s\"" % (rid, when, worst, what, excerpt)


def _id_from_label(label):
    """Recover the record id (first token) from a combo label."""
    if not label or label == _NO_RECORDS:
        return ""
    return label.split(" ", 1)[0].strip()


_NO_RECORDS = "no records yet"

# Strictly-increasing token for IS_CHANGED: each call returns a distinct value so
# ComfyUI's execution cache never skips a repeat queue (see IS_CHANGED below).
_IS_CHANGED_COUNTER = itertools.count()


class PreFlightOutcome:
    CATEGORY = CATEGORY
    FUNCTION = "log"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    DESCRIPTION = (
        "Record what a platform actually did to a published post — the feedback "
        "half of the loop. Pick the prediction from the list (newest first), the "
        "platform, and the result (clean / demoted / removed). No image input: "
        "outcomes attach to records, not files.\n\n"
        "The list is built when node definitions load, so new predictions appear "
        "after a browser refresh (or ComfyUI's Refresh button). record_id_override "
        "bypasses the list entirely — the id is shown in the Report summary.")

    @classmethod
    def INPUT_TYPES(cls):
        try:
            records = feedback.recent_predictions(feedback_store_path(), limit=50)
            labels = [_format_record_label(r) for r in records]
        except Exception:
            labels = []
        if not labels:
            labels = [_NO_RECORDS]
        return {
            "required": {
                "record": (labels, {"tooltip": "The prediction to attach an outcome "
                                              "to (newest first). Refresh to update."}),
                "platform": (list(feedback.PLATFORMS), {"default": "instagram",
                    "tooltip": "Which platform this outcome is for."}),
                "result": (list(feedback.RESULTS), {"default": "clean",
                    "tooltip": "clean = normal reach; demoted = suppressed / FYF-"
                               "ineligible / flagged; removed = taken down. Pick one "
                               "fixed personal heuristic for 'demoted' and stick to it."}),
                "record_id_override": ("STRING", {"default": "",
                    "tooltip": "If set, this id is used instead of the list selection "
                               "(bypasses list staleness)."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always-different so a repeat queue with identical inputs still writes —
        # otherwise the execution cache would skip logging the second platform.
        # A monotonic counter is collision-free (unlike a same-tick time.time()).
        return next(_IS_CHANGED_COUNTER)

    def log(self, record, platform, result, record_id_override=""):
        record_id = record_id_override.strip() or _id_from_label(record)
        if not record_id:
            return ("no record selected — pick one from the list or set "
                    "record_id_override",)
        try:
            feedback.log_outcome(record_id, platform, result, path=feedback_store_path())
            return ("logged: %s %s=%s" % (record_id, platform, result),)
        except Exception as exc:  # never raise into the graph
            return ("error: %s" % exc,)


NODE_CLASS_MAPPINGS = {
    "PreFlightObserve": PreFlightObserve,
    "PreFlightReport": PreFlightReport,
    "PreFlightOutcome": PreFlightOutcome,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PreFlightObserve": "PreFlight: Observe (Qwen-VL)",
    "PreFlightReport": "PreFlight: Report (IG/TikTok/X)",
    "PreFlightOutcome": "PreFlight: Outcome (feedback)",
}
