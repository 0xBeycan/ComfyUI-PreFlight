"""
Tests for the append-only feedback store and the calibration report.

Plain pytest, no ComfyUI / torch. Each test writes to a fresh ``tmp_path`` store.
"""

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import calibrate  # noqa: E402
import feedback  # noqa: E402
import rules  # noqa: E402


def _obs(**kw):
    base = dict(
        subject_appears_under_18=False, garment="regular", setting="studio",
        framing="full_body", pose="neutral", see_through_or_wet=False,
        exposure="none", nudity_or_sexual_act=False, motion_flags="none",
        visible_text="", confidence="high", meta={"schema_version": "1"},
    )
    base.update(kw)
    return base


def _report(**kw):
    return rules.judge(_obs(**kw))


# --- round trip ---------------------------------------------------------------

def test_log_prediction_then_outcome_joins(tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = feedback.log_prediction(_report(garment="bikini", setting="beach_pool",
                                          exposure="mild"),
                                  caption="summer drop", path=store)
    assert len(rid) == 8
    feedback.log_outcome(rid, "instagram", "demoted", path=store)

    joined = feedback.load_joined(store)
    assert len(joined) == 1
    assert joined[0]["id"] == rid
    assert joined[0]["outcomes"] == {"instagram": "demoted"}
    assert joined[0]["caption_excerpt"] == "summer drop"
    assert "base.bikini" in joined[0]["fired_rules"]


def test_prediction_record_shape(tmp_path):
    store = tmp_path / "fb.jsonl"
    feedback.log_prediction(_report(garment="lingerie", exposure="significant"),
                            caption="x", path=store)
    line = store.read_text().strip()
    rec = json.loads(line)
    assert rec["type"] == "prediction"
    assert rec["engine_version"] == rules.ENGINE_VERSION
    assert rec["schema_version"] == "1"
    assert rec["ts"].endswith("Z")
    assert set(rec) >= {"type", "id", "ts", "engine_version", "schema_version",
                        "observations", "caption_excerpt", "caption_flags",
                        "fired_rules", "verdicts"}


def test_caption_excerpt_truncated_to_80(tmp_path):
    store = tmp_path / "fb.jsonl"
    long_caption = "z" * 200
    feedback.log_prediction(_report(), caption=long_caption, path=store)
    rec = json.loads(store.read_text().strip())
    assert rec["caption_excerpt"] == "z" * 80


# --- append-only semantics ----------------------------------------------------

def test_last_write_wins_per_id_platform(tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = feedback.log_prediction(_report(), path=store)
    feedback.log_outcome(rid, "instagram", "clean", path=store)
    feedback.log_outcome(rid, "instagram", "removed", path=store)   # later wins
    joined = feedback.load_joined(store)
    assert joined[0]["outcomes"]["instagram"] == "removed"
    # every write is a distinct line (nothing updated in place)
    assert len(store.read_text().strip().splitlines()) == 3


def test_multiple_platforms_attach_independently(tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = feedback.log_prediction(_report(), path=store)
    feedback.log_outcome(rid, "instagram", "demoted", path=store)
    feedback.log_outcome(rid, "tiktok", "removed", path=store)
    outcomes = feedback.load_joined(store)[0]["outcomes"]
    assert outcomes == {"instagram": "demoted", "tiktok": "removed"}


def test_predictions_preserve_first_seen_order(tmp_path):
    store = tmp_path / "fb.jsonl"
    ids = [feedback.log_prediction(_report(), path=store) for _ in range(3)]
    assert [p["id"] for p in feedback.load_joined(store)] == ids


# --- recent_predictions (Outcome combo source) --------------------------------

def test_recent_predictions_newest_first_and_limited(tmp_path):
    store = tmp_path / "fb.jsonl"
    ids = [feedback.log_prediction(_report(), path=store) for _ in range(5)]
    recent = feedback.recent_predictions(store, limit=3)
    assert [p["id"] for p in recent] == list(reversed(ids))[:3]


# --- robustness ---------------------------------------------------------------

def test_missing_store_reads_empty(tmp_path):
    assert feedback.load_joined(tmp_path / "nope.jsonl") == []
    assert feedback.recent_predictions(tmp_path / "nope.jsonl") == []


def test_corrupt_line_is_skipped(tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = feedback.log_prediction(_report(), path=store)
    with store.open("a") as fh:
        fh.write("this is not json\n")
    assert len(feedback.load_joined(store)) == 1
    assert feedback.load_joined(store)[0]["id"] == rid


@pytest.mark.parametrize("platform,result", [
    ("myspace", "clean"),
    ("instagram", "shadowbanned"),
])
def test_log_outcome_rejects_bad_values(tmp_path, platform, result):
    store = tmp_path / "fb.jsonl"
    with pytest.raises(ValueError):
        feedback.log_outcome("abcd1234", platform, result, path=store)


# --- calibration report -------------------------------------------------------

def test_calibrate_report_counts_and_rules(tmp_path):
    store = tmp_path / "fb.jsonl"
    # Bikini-at-beach predicted RISK on IG; three came back clean -> over-predicted.
    for _ in range(3):
        rid = feedback.log_prediction(
            _report(garment="bikini", setting="beach_pool", exposure="mild"), path=store)
        feedback.log_outcome(rid, "instagram", "clean", path=store)
    # A prediction with no outcome must be ignored by calibration (costs nothing).
    feedback.log_prediction(_report(nudity_or_sexual_act=True, exposure="significant"),
                            path=store)

    report = calibrate.build_report(feedback.load_joined(store))
    assert "Per-platform calibration" in report
    assert "Per-rule outcomes" in report
    assert "base.bikini" in report
    assert "predictions: 4" in report and "with at least one outcome: 3" in report


def test_calibrate_empty_store(tmp_path):
    report = calibrate.build_report(feedback.load_joined(tmp_path / "empty.jsonl"))
    assert "Store is empty" in report
