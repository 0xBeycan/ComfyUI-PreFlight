"""
PreFlight rules engine — maps sensor observations to per-platform verdict ranges.

Pure stdlib (``re``, ``json``). No ComfyUI / torch / transformers imports: this
module is the single home of the rule set and must be importable by external
services outside ComfyUI. All platform logic lives here; the ComfyUI node layer
only calls ``judge`` / ``summary_text`` and never re-implements a rule.

The model is a pure observation sensor; this engine is the only place that
"judges". Every verdict is a RANGE (best..worst) because the sensor cannot
resolve everything with certainty and because some rules depend on factors
outside the frame (setting, region, account signals). ``range_drivers`` names
the unknown so the user can collapse the range themselves.

See the functional spec §6 for the authoritative rule tables. Every rule has a
stable ID (e.g. ``base.bikini``); whenever a rule changes a verdict its ID is
appended to ``fired_rules`` in application order — that is the hook calibrate.py
uses to attribute real-world outcomes to individual rules.
"""

import re

ENGINE_VERSION = "1.0.0"

# Observation schema versions this engine knows how to interpret. Stage 0 fails
# closed on anything else, so an observation produced by a newer/older schema is
# never silently judged against rules written for a different shape.
KNOWN_SCHEMA_VERSIONS = {"1"}

# --- severity ----------------------------------------------------------------
# Internal severity is an int so bumps/merges are plain saturating arithmetic.
OK, RISK, BLOCK = 0, 1, 2
_STATUS = {OK: "OK", RISK: "RISK", BLOCK: "BLOCK"}
_ICON = {"OK": "✅", "RISK": "⚠️", "BLOCK": "❌", "UNKNOWN": "❓"}

PLATFORMS = ("instagram", "tiktok", "x")

DISCLAIMER = (
    "This is a prediction, not a platform approval. Account history, bio "
    "signals, posting cadence and cluster patterns are not considered. Final "
    "judgment is yours."
)
_FAIL_CLOSED_REASON = "vision sensor unavailable or incompatible — manual review required"

# Ordered so ``exposure >= moderate`` is a simple integer comparison.
_EXPOSURE_ORDER = {"none": 0, "mild": 1, "moderate": 2, "significant": 3}
_MINIMAL_GARMENTS = {"lingerie", "underwear_only"}
_MILD_GARMENTS = {"miniskirt", "croptop", "shorts", "fitness_wear"}
_FRAMING_SEXUALIZED = {"chest_focus", "butt_focus", "lowangle_body"}
_FLAGGED_EMOJI = {"🍑", "🍆", "💦", "👅", "🥵"}


def _clamp(n):
    """Saturate a severity to the valid range [OK, BLOCK]."""
    return max(OK, min(BLOCK, n))


# ---------------------------------------------------------------------------
# Caption / in-image text flags (§6.1)
# ---------------------------------------------------------------------------

# Ordered (name, compiled pattern). Patterns that can feed a hard rule must be
# high-precision — a bare "dm for" must NOT match ("DM for collabs"); only the
# content-qualified forms count.
_FLAG_PATTERNS = [
    ("adult_platform_mention", re.compile(r"(?i)(onlyfans|only\s*fans|fanvue|fansly)")),
    ("price_or_subscription_cta", re.compile(
        r"(?i)(dm\s*for\s*(more|content|pics?|vids?|access|fun)|pay\s*to|"
        r"subscribe|sub\s*for|ppv|exclusive\s*content)")),
    ("age_marker", re.compile(r"(?i)(18\s*\+|\bnsfw\b|\bspicy\s+(content|pics?|vids?|link)\b)")),
    ("adult_link", re.compile(r"(?i)(fanvue\.com|onlyfans\.com|fansly\.com)")),
]


def caption_flags(text):
    """Deterministic flags present in a single text blob (pure function).

    Returns flag names in a stable order. ``flagged_emoji`` is membership-based;
    the rest are regex. No estimation happens here — this is exact matching.
    """
    if not isinstance(text, str) or not text:
        return []
    flags = [name for name, pat in _FLAG_PATTERNS if pat.search(text)]
    if any(ch in _FLAGGED_EMOJI for ch in text):
        flags.append("flagged_emoji")
    return flags


# ---------------------------------------------------------------------------
# Verdict helpers — a verdict is {"best":int,"worst":int,"hard":bool,
# "reasons":[str]} (+ "label_required":bool for X).
# ---------------------------------------------------------------------------

def _new_verdict():
    return {"best": OK, "worst": OK, "hard": False, "reasons": []}


def _add_reason(v, reason):
    if reason and reason not in v["reasons"]:
        v["reasons"].append(reason)


def _normalize(v):
    """Clamp both bounds and re-enforce the best <= worst invariant.

    Called after every stage. If a stage raised ``best`` above ``worst`` (e.g.
    raising IG best to RISK on a formerly OK/OK verdict), the range collapses up
    to a point rather than becoming the impossible [RISK, OK].
    """
    v["best"] = _clamp(v["best"])
    v["worst"] = _clamp(max(v["worst"], v["best"]))


def _merge(v, best, worst, reason=None):
    """Max-merge a (best, worst) contribution into a verdict (never lowers)."""
    v["best"] = max(v["best"], best)
    v["worst"] = max(v["worst"], worst)
    _add_reason(v, reason)


def _bump_worst(v, n=1):
    v["worst"] = _clamp(v["worst"] + n)


def _raise_best(v, floor):
    v["best"] = _clamp(max(v["best"], floor))


# ---------------------------------------------------------------------------
# Stage 2 — base rules
# ---------------------------------------------------------------------------

def _apply_base_rules(V, obs, fired):
    ig, tt, x = V["instagram"], V["tiktok"], V["x"]
    garment = obs.get("garment", "regular")
    exposure = obs.get("exposure", "none")
    exposure_rank = _EXPOSURE_ORDER.get(exposure, 0)
    setting = obs.get("setting", "indoor_other")
    matched = []

    if obs.get("nudity_or_sexual_act") is True:
        matched.append("base.nudity")
        _merge(ig, BLOCK, BLOCK, "nudity (violates community standards)")
        _merge(tt, BLOCK, BLOCK, "nudity")
        _merge(x, RISK, RISK, "adult label + sensitive media settings required")

    if garment in _MINIMAL_GARMENTS or exposure == "significant":
        matched.append("base.minimal")
        _merge(ig, RISK, RISK, "minimal coverage — suggestive demotion risk")
        _merge(tt, BLOCK, BLOCK, "significant exposure — FYF ineligible")
        _merge(x, RISK, RISK, "adult label recommended")

    if garment == "bikini":
        matched.append("base.bikini")
        _merge(ig, RISK, RISK, "bikini — demotion possible")
        # TikTok collapses on setting: FYF-killed indoors, RISK at beach/pool,
        # otherwise the RISK..BLOCK range stands (setting is the unknown).
        if setting in ("bedroom", "indoor_other"):
            _merge(tt, BLOCK, BLOCK, "bikini indoors — FYF ineligible")
        elif setting == "beach_pool":
            _merge(tt, RISK, RISK, "bikini at beach/pool — demotion possible")
        else:
            _merge(tt, RISK, BLOCK, "bikini — RISK at beach/pool, FYF kill indoors")
        # X: OK/RISK, label only when exposure is at least moderate.
        _merge(x, OK, RISK, "bikini")

    if obs.get("see_through_or_wet") is True:
        matched.append("base.see_through")
        _merge(ig, BLOCK, BLOCK, "see-through — explicit demotion example in IG guidelines")
        _merge(tt, BLOCK, BLOCK, "implied nudity / age-restricted")
        _merge(x, RISK, RISK, "see-through / wet — adult label recommended")

    if exposure == "moderate":
        matched.append("base.exposure_mod")
        _merge(ig, RISK, RISK, "moderate exposure (sideboob/underboob/partial buttock)")
        _merge(tt, RISK, BLOCK, "moderate exposure is restricted in some regions")
        _merge(x, RISK, RISK, "moderate exposure — adult label recommended")

    if exposure == "mild" or garment in _MILD_GARMENTS:
        matched.append("base.exposure_mild")
        _merge(ig, OK, RISK, "mild exposure / revealing casualwear")
        _merge(tt, OK, RISK, "mild exposure / revealing casualwear")
        _merge(x, OK, OK)

    if not matched:
        matched.append("base.default")

    fired.extend(matched)


# ---------------------------------------------------------------------------
# Stage 3 — modifiers (saturating bumps, applied in listed order)
# ---------------------------------------------------------------------------

def _apply_modifiers(V, obs, flags, motion, fired):
    ig, tt = V["instagram"], V["tiktok"]

    if obs.get("framing") in _FRAMING_SEXUALIZED:
        fired.append("mod.framing")
        _bump_worst(ig)
        _bump_worst(tt)
        note = "sexualized framing (demoted even with ordinary clothing)"
        _add_reason(ig, note)
        _add_reason(tt, note)

    pose = obs.get("pose", "neutral")
    if pose == "suggestive":
        fired.append("mod.pose_suggestive")
        _bump_worst(ig)
        _bump_worst(tt)
        _raise_best(ig, RISK)
        _add_reason(ig, "IG demotes suggestive poses directly")
        _add_reason(tt, "suggestive pose")
    elif pose == "mildly_suggestive":
        fired.append("mod.pose_mild")
        _bump_worst(tt)
        _add_reason(tt, "mildly suggestive pose")

    if motion == "twerk_grind_striptease":
        fired.append("mod.motion_twerk")
        # Floor (max-merge, not assignment) so an already-worse verdict is kept.
        _merge(ig, BLOCK, BLOCK, "twerk/grind/striptease motion")
        _merge(tt, BLOCK, BLOCK, "twerk/grind/striptease motion — age-restricted")
        _merge(V["x"], RISK, RISK, "sexualized motion — adult label recommended")
    elif motion == "intimate_kissing":
        fired.append("mod.motion_kiss")
        _bump_worst(tt)
        _add_reason(tt, "intimate kissing")

    if "age_marker" in flags or "flagged_emoji" in flags:
        fired.append("mod.text_age")
        _bump_worst(ig)
        _bump_worst(tt)
        note = "adult/age-coded text or emoji in caption or image"
        _add_reason(ig, note)
        _add_reason(tt, note)


# ---------------------------------------------------------------------------
# Stage 4 — hard collapses (immune to Stage 5 widening)
# ---------------------------------------------------------------------------

def _apply_hard_rules(V, obs, flags, fired):
    ig, tt = V["instagram"], V["tiktok"]
    garment = obs.get("garment", "regular")
    exposure = obs.get("exposure", "none")
    pose = obs.get("pose", "neutral")

    solicit = any(f in flags for f in
                  ("adult_platform_mention", "price_or_subscription_cta", "adult_link"))
    suggestive_visual = (exposure != "none" or pose != "neutral"
                         or garment in ({"bikini"} | _MINIMAL_GARMENTS))
    if solicit and suggestive_visual:
        fired.append("hard.solicitation")
        _merge(ig, BLOCK, BLOCK,
               "sexual solicitation: suggestive visual + caption offer = removal")
        ig["hard"] = True

    if "adult_link" in flags:
        fired.append("hard.adult_link")
        note = ("adult platform link (caption or in-image watermark); "
                "applies independently of the visual")
        _merge(ig, BLOCK, BLOCK, note)
        _merge(tt, BLOCK, BLOCK, note)
        ig["hard"] = True
        tt["hard"] = True


# ---------------------------------------------------------------------------
# Stage 5 — confidence widening (never reopens a hard collapse)
# ---------------------------------------------------------------------------

def _apply_confidence(V, obs, fired):
    if obs.get("confidence") != "low":
        return
    widened = False
    for v in V.values():
        if v["hard"]:
            continue  # widening must never reopen a hard verdict
        v["best"] = _clamp(v["best"] - 1)
        v["worst"] = _clamp(v["worst"] + 1)
        _add_reason(v, "low sensor confidence — range widened")
        widened = True
    if widened:
        fired.append("mod.confidence_low")


# ---------------------------------------------------------------------------
# Stage 6 — X label requirement
# ---------------------------------------------------------------------------

def _apply_x_label(V, obs, motion):
    exposure = obs.get("exposure", "none")
    required = (
        obs.get("nudity_or_sexual_act") is True
        or exposure in ("moderate", "significant")
        or obs.get("see_through_or_wet") is True
        or obs.get("garment") in _MINIMAL_GARMENTS
        or obs.get("pose") == "suggestive"
        or motion == "twerk_grind_striptease"
    )
    x = V["x"]
    x["label_required"] = bool(required)
    if required:
        _add_reason(x, "never use as profile photo, header, banner, or live thumbnail")


# ---------------------------------------------------------------------------
# range_drivers — human-readable explanations of what the range hinges on
# ---------------------------------------------------------------------------

def _range_drivers(obs, fired, widened_low_conf):
    drivers = []
    if "base.bikini" in fired:
        setting = obs.get("setting", "indoor_other")
        if setting == "beach_pool":
            drivers.append(
                "TikTok: bikini reported at 'beach_pool' — FYF demotion (RISK). If it "
                "were indoors (bedroom/indoor_other) TikTok would be FYF-ineligible (BLOCK).")
        elif setting in ("bedroom", "indoor_other"):
            drivers.append(
                "TikTok: bikini reported indoors ('%s') — FYF-ineligible (BLOCK). If "
                "this is actually a beach or pool, the verdict drops to RISK." % setting)
        else:
            drivers.append(
                "TikTok: bikini reported in '%s' — verdict spans RISK (beach/pool) to "
                "BLOCK (indoors); confirm the setting to collapse it." % setting)
    if "base.exposure_mod" in fired:
        drivers.append(
            "TikTok: moderate exposure (sideboob/underboob/partial buttock) is "
            "region-restricted — RISK in most regions, BLOCK where enforcement is stricter.")
    if widened_low_conf:
        drivers.append(
            "All non-final verdicts were widened because the sensor reported LOW "
            "confidence — a clearer, less-occluded image would tighten these ranges.")
    return drivers


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize(v, is_x=False):
    out = {
        "best": _STATUS[v["best"]],
        "worst": _STATUS[v["worst"]],
        "hard": bool(v["hard"]),
        "reasons": list(v["reasons"]),
    }
    if is_x:
        out["label_required"] = bool(v.get("label_required", False))
    return out


def _build_report(observations, flags, fired, verdicts, drivers, unknown):
    return {
        "engine_version": ENGINE_VERSION,
        "record_id": "",  # injected by the Report node after logging (§7.3)
        "observations": observations if isinstance(observations, dict) else {},
        "caption_flags": flags,
        "fired_rules": fired,
        "verdicts": {
            "instagram": _serialize(verdicts["instagram"]),
            "tiktok": _serialize(verdicts["tiktok"]),
            "x": _serialize(verdicts["x"], is_x=True),
        },
        "range_drivers": drivers,
        "unknown": unknown,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge(observations, caption_text="", is_video=False):
    """Apply the full pipeline and return a structured report (§6.3).

    Stages run in strict order and must not be reordered: 0 fail-closed,
    1 minor override, 2 base rules, 3 modifiers, 4 hard collapses, 5 confidence
    widening, 6 X label.
    """
    # Flags are computed over the UNION of the caption and the in-image text, so
    # a fanvue.com watermark burned into the frame fires the same rules a caption
    # link would. caption_flags() itself stays a pure single-text function.
    visible_text = ""
    if isinstance(observations, dict):
        vt = observations.get("visible_text", "")
        visible_text = vt if isinstance(vt, str) else ""
    flags = caption_flags((caption_text or "") + "\n" + visible_text)

    # --- Stage 0: fail closed -------------------------------------------------
    schema_version = None
    if isinstance(observations, dict):
        meta = observations.get("meta")
        if isinstance(meta, dict):
            schema_version = meta.get("schema_version")
    if (observations is None or not isinstance(observations, dict)
            or "error" in observations
            or schema_version not in KNOWN_SCHEMA_VERSIONS):
        unknown_v = {"best": "UNKNOWN", "worst": "UNKNOWN", "hard": False,
                     "reasons": [_FAIL_CLOSED_REASON]}
        verdicts = {
            "instagram": dict(unknown_v),
            "tiktok": dict(unknown_v),
            "x": {**unknown_v, "label_required": False},
        }
        # Rebuild reasons as independent lists (dict() shares the list object).
        for name in PLATFORMS:
            verdicts[name]["reasons"] = [_FAIL_CLOSED_REASON]
        return {
            "engine_version": ENGINE_VERSION,
            "record_id": "",
            "observations": observations if isinstance(observations, dict) else {},
            "caption_flags": flags,
            "fired_rules": ["guard.fail_closed"],
            "verdicts": verdicts,
            "range_drivers": [],
            "unknown": True,
            "disclaimer": DISCLAIMER,
        }

    # --- Stage 1: global minor override --------------------------------------
    if observations.get("subject_appears_under_18") is True:
        verdicts = {}
        for name in PLATFORMS:
            verdicts[name] = {"best": "BLOCK", "worst": "BLOCK", "hard": True,
                              "reasons": ["possible minor"]}
        verdicts["x"]["label_required"] = False
        return {
            "engine_version": ENGINE_VERSION,
            "record_id": "",
            "observations": observations,
            "caption_flags": flags,
            "fired_rules": ["override.minor"],
            "verdicts": verdicts,
            "range_drivers": [],
            "unknown": False,
            "disclaimer": DISCLAIMER,
        }

    # motion flags only count for genuine multi-frame (video) input; guard a
    # stray flag on a single still.
    motion = observations.get("motion_flags", "none") if is_video else "none"

    V = {name: _new_verdict() for name in PLATFORMS}
    fired = []

    _apply_base_rules(V, observations, fired)
    for v in V.values():
        _normalize(v)

    _apply_modifiers(V, observations, flags, motion, fired)
    for v in V.values():
        _normalize(v)

    _apply_hard_rules(V, observations, flags, fired)
    for v in V.values():
        _normalize(v)

    _low_conf = observations.get("confidence") == "low"
    _apply_confidence(V, observations, fired)
    widened = _low_conf and "mod.confidence_low" in fired
    for v in V.values():
        _normalize(v)

    _apply_x_label(V, observations, motion)

    drivers = _range_drivers(observations, fired, widened)
    return _build_report(observations, flags, fired, V, drivers, unknown=False)


# ---------------------------------------------------------------------------
# Human-readable summary (§6.4)
# ---------------------------------------------------------------------------

def _status_part(best, worst):
    if best == worst:
        return "%s %s" % (_ICON[best], best)
    return "%s %s → %s %s" % (_ICON[best], best, _ICON[worst], worst)


def summary_text(report):
    """One-screen preview of a report (§6.4)."""
    verdicts = report.get("verdicts", {})
    labels = [("Instagram", "instagram"), ("TikTok", "tiktok"), ("X", "x")]
    lines = []
    for label, key in labels:
        v = verdicts.get(key, {})
        best = v.get("best", "UNKNOWN")
        worst = v.get("worst", "UNKNOWN")
        status = _status_part(best, worst)
        if key == "x" and v.get("label_required"):
            reason = "adult label recommended"
        else:
            reason = "; ".join(v.get("reasons", []))
        line = "%-11s%s" % (label + ":", status)
        if reason:
            line += " — " + reason
        lines.append(line)

    for driver in report.get("range_drivers", []):
        lines.append("")
        lines.append("⚡ Range driver: " + driver)

    record_id = report.get("record_id", "")
    if record_id:
        lines.append("")
        lines.append("record: " + record_id)

    return "\n".join(lines)
