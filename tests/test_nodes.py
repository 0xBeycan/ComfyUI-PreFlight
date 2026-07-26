"""
Tests for the ComfyUI node layer, with the model fully mocked.

Runs with pytest + numpy + Pillow (no torch, no ComfyUI). The GPU path is
replaced by a stub ``run_observation``; everything else — JSON extraction, the
single retry, meta injection, frame sampling, the feedback wiring, and graceful
degradation — is exercised for real.
"""

import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import feedback  # noqa: E402
import nodes  # noqa: E402
import rules  # noqa: E402

VALID_JSON = json.dumps({
    "subject_appears_under_18": False, "garment": "bikini", "setting": "beach_pool",
    "framing": "full_body", "pose": "neutral", "see_through_or_wet": False,
    "exposure": "mild", "nudity_or_sexual_act": False, "motion_flags": "none",
    "visible_text": "", "confidence": "high",
})


def _obs_json(**kw):
    base = dict(
        subject_appears_under_18=False, garment="regular", setting="studio",
        framing="full_body", pose="neutral", see_through_or_wet=False,
        exposure="none", nudity_or_sexual_act=False, motion_flags="none",
        visible_text="", confidence="high", meta={"schema_version": "1",
        "prompt_version": "1", "model_name": "test", "frames_analyzed": 1})
    base.update(kw)
    return json.dumps(base)


# --- _extract_json ------------------------------------------------------------

@pytest.mark.parametrize("raw,ok", [
    ('{"a": 1}', True),
    ('```json\n{"a": 1}\n```', True),                    # fenced
    ('Sure! Here you go:\n{"a": 1}\nHope that helps', True),  # prose around it
    ('{"a": {"b": 2}, "c": 3}', True),                   # nested
    ('{"text": "has } brace in string"}', True),         # brace inside string
    ('no json here', False),
    ('', False),
    (None, False),
])
def test_extract_json(raw, ok):
    result = nodes._extract_json(raw)
    assert (result is not None) == ok


def test_extract_json_brace_in_string_value():
    r = nodes._extract_json('{"visible_text": "50% off {sale}"}')
    assert r == {"visible_text": "50% off {sale}"}


# --- _observe_from_generate: retry + meta -------------------------------------

def test_observe_valid_first_try_single_call():
    calls = []

    def gen(extra):
        calls.append(extra)
        return VALID_JSON

    obs, raw = nodes._observe_from_generate(gen, "Qwen3-VL-8B-Instruct", 6)
    assert calls == [""]                                 # exactly one call, no retry
    assert "error" not in obs
    assert obs["garment"] == "bikini"
    assert obs["meta"] == {"schema_version": nodes.prompts.SCHEMA_VERSION,
                           "prompt_version": nodes.prompts.PROMPT_VERSION,
                           "model_name": "Qwen3-VL-8B-Instruct", "frames_analyzed": 6}


def test_observe_retries_once_then_succeeds():
    outs = iter(["not json at all", "```json\n" + VALID_JSON + "\n```"])
    calls = []

    def gen(extra):
        calls.append(extra)
        return next(outs)

    obs, raw = nodes._observe_from_generate(gen, "m", 1)
    assert len(calls) == 2 and calls[1] == nodes._RETRY_INSTRUCTION
    assert obs["garment"] == "bikini"
    assert "--- retry ---" in raw                        # both responses kept for debug


def test_observe_two_failures_returns_error_no_meta():
    calls = []

    def gen(extra):
        calls.append(extra)
        return "still not json"

    obs, raw = nodes._observe_from_generate(gen, "m", 1)
    assert len(calls) == 2                               # exactly one retry, then stop
    assert "error" in obs and "meta" not in obs          # error obs -> rules fail closed


def test_observe_coerces_unknown_enum():
    bad = json.dumps({**json.loads(VALID_JSON), "exposure": "extreme"})
    obs, _ = nodes._observe_from_generate(lambda e: bad, "m", 1)
    assert obs["exposure"] == "moderate"                 # coerced to cautious default


# --- full observe() with a stubbed model --------------------------------------

def _patch_model(monkeypatch, run_impl):
    monkeypatch.setattr(nodes.model_manager, "load_model",
                        lambda *a, **k: (object(), object()))
    monkeypatch.setattr(nodes.model_manager, "run_observation", run_impl)
    monkeypatch.setattr(nodes.model_manager, "release", lambda *a, **k: None)


def test_observe_node_happy_path(monkeypatch):
    seen = {"calls": 0, "nframes": None}

    def run_impl(model, processor, pil_frames, prompt, **kw):
        seen["calls"] += 1
        seen["nframes"] = len(pil_frames)
        return VALID_JSON

    _patch_model(monkeypatch, run_impl)
    image = np.random.rand(10, 8, 8, 3).astype(np.float32)   # 10-frame batch
    obs_json, raw = nodes.PreFlightObserve().observe(
        image, "Qwen3-VL-8B-Instruct", "8-bit", "auto", 6, True, "auto")

    obs = json.loads(obs_json)
    assert obs["garment"] == "bikini"
    assert obs["meta"]["frames_analyzed"] == 6
    assert seen["calls"] == 1                            # ONE model call for the batch
    assert seen["nframes"] == 6                          # sampled down to max_frames


def test_observe_node_never_raises_on_model_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    _patch_model(monkeypatch, boom)
    image = np.random.rand(1, 8, 8, 3).astype(np.float32)
    obs_json, raw = nodes.PreFlightObserve().observe(
        image, "Qwen3-VL-8B-Instruct", "8-bit", "auto", 6, True, "auto")
    obs = json.loads(obs_json)
    assert "error" in obs and "CUDA out of memory" in obs["error"]

    # And that error observation makes Report fail closed to UNKNOWN, never OK.
    report_json, summary, rid, _ = nodes.PreFlightReport().report(obs_json, "", False)
    assert json.loads(report_json)["unknown"] is True


# --- Report node --------------------------------------------------------------

def test_report_logs_and_injects_record_id(monkeypatch, tmp_path):
    store = tmp_path / "fb.jsonl"
    monkeypatch.setattr(nodes, "feedback_store_path", lambda: store)
    report_json, summary, rid, img = nodes.PreFlightReport().report(
        _obs_json(garment="bikini", setting="bedroom", exposure="mild"),
        caption="hi", log_prediction=True, image="IMG")

    assert len(rid) == 8
    assert json.loads(report_json)["record_id"] == rid   # injected into JSON
    assert ("record: " + rid) in summary                 # and into the summary
    assert img == "IMG"                                   # pass-through
    assert len(store.read_text().strip().splitlines()) == 1   # exactly one line


def test_report_logging_off_writes_nothing(monkeypatch, tmp_path):
    store = tmp_path / "fb.jsonl"
    monkeypatch.setattr(nodes, "feedback_store_path", lambda: store)
    report_json, summary, rid, _ = nodes.PreFlightReport().report(
        _obs_json(), caption="", log_prediction=False)
    assert rid == ""
    assert "record:" not in summary
    assert not store.exists()


def test_report_survives_store_failure(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(nodes.feedback, "log_prediction", boom)
    monkeypatch.setattr(nodes, "feedback_store_path", lambda: tmp_path / "fb.jsonl")
    # Must not raise; verdicts still returned, record_id empty (§9.12).
    report_json, summary, rid, _ = nodes.PreFlightReport().report(
        _obs_json(garment="lingerie", exposure="significant"), "", True)
    assert rid == ""
    assert json.loads(report_json)["verdicts"]["tiktok"]["worst"] == "BLOCK"


# --- Outcome node -------------------------------------------------------------

def test_record_label_roundtrips_to_id():
    rec = {"id": "a3f9c2d1", "ts": "2026-07-24T14:02:11Z",
           "verdicts": {"instagram": {"worst": "RISK"}, "tiktok": {"worst": "BLOCK"},
                        "x": {"worst": "OK"}},
           "observations": {"garment": "bikini", "setting": "beach_pool"},
           "caption_excerpt": "summer drop 🌞 limited"}
    label = nodes._format_record_label(rec)
    assert label.startswith("a3f9c2d1 ")
    assert "IG:RISK" in label and "TT:BLOCK" in label
    assert "bikini/beach_pool" in label
    assert nodes._id_from_label(label) == "a3f9c2d1"


def test_outcome_logs_via_override(monkeypatch, tmp_path):
    store = tmp_path / "fb.jsonl"
    monkeypatch.setattr(nodes, "feedback_store_path", lambda: store)
    rid = feedback.log_prediction(rules.judge(json.loads(_obs_json())), path=store)

    (status,) = nodes.PreFlightOutcome().log(
        record=nodes._NO_RECORDS, platform="tiktok", result="removed",
        record_id_override=rid)
    assert status == "logged: %s tiktok=removed" % rid
    assert feedback.load_joined(store)[0]["outcomes"] == {"tiktok": "removed"}


def test_outcome_no_selection_reports_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "feedback_store_path", lambda: tmp_path / "fb.jsonl")
    (status,) = nodes.PreFlightOutcome().log(
        record=nodes._NO_RECORDS, platform="instagram", result="clean",
        record_id_override="")
    assert "no record selected" in status


def test_outcome_bad_value_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "feedback_store_path", lambda: tmp_path / "fb.jsonl")
    (status,) = nodes.PreFlightOutcome().log(
        record=nodes._NO_RECORDS, platform="instagram", result="clean",
        record_id_override="abcd1234")
    # abcd1234 is a fine id; force an error via a bad platform instead:
    (status2,) = nodes.PreFlightOutcome().log(
        record="badlabel", platform="myspace", result="clean", record_id_override="x")
    assert status2.startswith("error:")


def test_outcome_is_changed_always_differs():
    a = nodes.PreFlightOutcome.IS_CHANGED()
    b = nodes.PreFlightOutcome.IS_CHANGED()
    assert a != b                                        # cache-buster (§9.11)


# --- package mappings ---------------------------------------------------------

def test_node_mappings_complete():
    assert set(nodes.NODE_CLASS_MAPPINGS) == {
        "PreFlightObserve", "PreFlightReport", "PreFlightOutcome"}
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["PreFlightObserve"] == \
        "PreFlight: Observe (Qwen-VL)"
    for cls in nodes.NODE_CLASS_MAPPINGS.values():
        assert cls.CATEGORY == "PreFlight"
        assert hasattr(cls, "INPUT_TYPES") and hasattr(cls, cls.FUNCTION)
