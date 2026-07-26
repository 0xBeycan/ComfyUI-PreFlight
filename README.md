# ComfyUI-PreFlight

Predict how a generated image or video would be treated by **Instagram, TikTok and X — before you publish it.** PreFlight flags two distinct risks per platform:

- **Hard limit** — removal (violates community standards).
- **Soft limit** — reach suppression (Instagram recommendation exclusion, TikTok For-You-Feed ineligibility).

It runs a local **Qwen-VL** model as a *pure observation sensor* — it describes what is visually present, it does **not** judge — and feeds those observations into a **deterministic rules engine** that maps them to per-platform verdict ranges. A feedback loop lets you record what the platform actually did, so the rules can be calibrated against reality over time.

> **This is an assistive signal, not an approval gate.** It never says "approved" or "rejected". Verdicts are ranges (best…worst); the report names the unknown that drives each range so you can collapse it yourself. Final judgment is yours.

---

## Why three nodes

| Node | Class | What it does |
|---|---|---|
| **PreFlight: Observe (Qwen-VL)** | `PreFlightObserve` | Runs Qwen-VL, returns a structured **observations JSON** (garment, exposure, framing, pose, in-image text, …). |
| **PreFlight: Report (IG/TikTok/X)** | `PreFlightReport` | Applies the rules engine, returns **per-platform verdicts** + a human summary, and logs the prediction. |
| **PreFlight: Outcome (feedback)** | `PreFlightOutcome` | Records what a platform actually did to a published post (days later). |

The split is deliberate: when a verdict looks wrong you need to know whether **perception** failed (the model misread the garment) or **rule application** failed. Keeping observations as a separate artifact also lets you re-evaluate old content against updated rules **without re-running the model** — every observation is stamped with the schema/prompt version that produced it.

---

## Install

1. Clone into your ComfyUI custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/0xBeycan/ComfyUI-PreFlight
   ```
2. Install dependencies (from the pack folder):
   ```bash
   pip install -r requirements.txt
   ```
   `transformers`, `accelerate`, `bitsandbytes`, `huggingface_hub`, `pillow`, `numpy`. **torch is not installed here** — it comes from your ComfyUI runtime.
3. Restart ComfyUI. The three nodes appear under the **`PreFlight`** category.

The model downloads automatically on first use (see below). No manual model setup is required.

---

## Models & the shared directory

Models live in **`ComfyUI/models/LLM/Qwen-VL/<model_name>`**, resolved via `folder_paths`. This is the same location other Qwen-VL packs use, so **a model already downloaded by another pack is reused, not downloaded again** — these are multi-GB files. PreFlight checks both `LLM` and `llm` casing before deciding a model is missing.

If the selected model is absent, PreFlight downloads it from Hugging Face (`snapshot_download`) into that shared directory, with progress in the ComfyUI progress bar and console.

Only **Instruct** variants are listed — "Thinking" variants emit reasoning text that breaks strict JSON output. Add a model by editing `models.json` (no code change needed).

### VRAM guidance

Approximate VRAM by model and quantization. Default is **Qwen3-VL-8B-Instruct at 8-bit**; drop to 4B or 2B for smaller GPUs.

| Model | Full (FP16) | 8-bit *(default)* | 4-bit |
|---|---|---|---|
| Qwen3-VL-8B-Instruct *(default)* | ~12 GB | ~7 GB | ~4.5 GB |
| Qwen3-VL-4B-Instruct | ~6 GB | ~3.5 GB | ~2 GB |
| Qwen3-VL-2B-Instruct | ~4 GB | ~2.5 GB | ~1.5 GB |

4-bit / 8-bit use **bitsandbytes** (CUDA). Quantized and CPU paths force `sdpa` attention automatically.

---

## Node reference

### PreFlight: Observe (Qwen-VL)

| Input | Type | Default | Notes |
|---|---|---|---|
| `image` | IMAGE | — | Single image or a video frame batch. |
| `model_name` | combo | `Qwen3-VL-8B-Instruct` | From `models.json`. |
| `quantization` | combo | `8-bit` | `4-bit` / `8-bit` / `None (FP16)`. |
| `attention_mode` | combo | `auto` | `auto` tries flash-attention-2, falls back to `sdpa`. |
| `max_frames` | INT | 6 | 1–16, sampled evenly from the batch, sent in **one** model call. |
| `keep_model_loaded` | BOOLEAN | True | Keep the model resident in VRAM between runs. |
| `device` | combo | `auto` | `auto` / `cuda` / `cpu`. |

**Outputs:** `observations_json` (STRING), `raw_response` (STRING, for debugging).

Generation is **deterministic** (greedy decoding, fixed seed, no sampling): the same image + settings produce byte-identical output run to run — a hard requirement for calibration. Errors never break the graph: a problem comes back as `{"error": ...}` so Report can fail closed.

### PreFlight: Report (IG/TikTok/X)

| Input | Type | Default | Notes |
|---|---|---|---|
| `observations_json` | STRING (forceInput) | — | From Observe. |
| `caption` | STRING (multiline) | `""` | Optional caption to be published. |
| `image` | IMAGE (optional) | — | Pass-through so the node can sit inline. |
| `log_prediction` | BOOLEAN | True | Append this prediction to the feedback store. |

**Outputs:** `report_json` (full structured report), `summary` (human-readable preview), `record_id` (feedback id; `""` when logging is off/failed), `image` (pass-through).

The caption **and** any text the model read inside the image (`visible_text`) are scanned together — so a `fanvue.com` watermark burned into a frame flags exactly like a caption link would.

#### Reading a verdict

Every verdict is a **range**, `best → worst`, plus severity meanings:

- **OK** ✅ — no expected problem.
- **RISK** ⚠️ — **reach demotion** (Instagram recommendation exclusion, general down-ranking). The post stays up but gets less reach.
- **BLOCK** ❌ — **removal** (IG/TikTok) or **TikTok For-You-Feed ineligibility** — a hard limit.
- **UNKNOWN** ❓ — the sensor was unavailable/incompatible; judge for yourself.

So `OK → RISK` means "probably fine, at worst demoted"; `RISK → BLOCK` means "demoted, and possibly removed depending on a named factor". `best` is the favourable case (clean account, benign context); `worst` is the cautious ceiling. Whenever `best ≠ worst`, a **range driver** names exactly what would collapse it (setting, region, sensor confidence, …). **BLOCK at `worst` is reserved for genuine policy triggers** — nudity, see-through, significant exposure, adult solicitation/links, explicit sexual motion, or a *sexualized* depiction of an apparent minor. Soft signals (framing, pose, mild exposure, suggestive text) top out at **RISK** — they demote, they don't remove.

> **On apparent minors:** the block only fires when an apparent minor is shown in a **sexualized** context (exposure, swim/intimate garment, suggestive pose/framing, sexual motion or text). A clothed, neutral subject is treated as ordinary content — a youthful-looking adult is not flagged as a minor.

### PreFlight: Outcome (feedback)

| Input | Type | Notes |
|---|---|---|
| `record` | combo | Recent predictions, newest first (see below). |
| `platform` | combo | `instagram` / `tiktok` / `x`. |
| `result` | combo | `clean` / `demoted` / `removed`. |
| `record_id_override` | STRING | When set, used instead of the combo selection. |

**Output:** `status` (STRING), e.g. `logged: a3f9c2d1 instagram=demoted`. Errors come back in the string; they never raise.

---

## Example workflows

- **`example_workflows/preflight_basic.json`** — Load Image → Observe → Report (with the image passed through to a Preview). Connect the `summary` / `report_json` STRING outputs to any text-display node to read them.
- **`example_workflows/preflight_outcome.json`** — a single Outcome node, meant to be opened days after posting.

> _Screenshot placeholder: add `example_workflows/preflight_basic.png` showing the graph._

---

## The feedback loop

The rules are **hypotheses** about platform behaviour. This is how they get corrected. Nothing here is ML and nothing self-tunes: predictions and real outcomes accumulate in a log, `calibrate.py` turns the log into per-rule statistics, and **a human** edits `rules.py` and bumps `ENGINE_VERSION`. Entering outcomes is optional — predictions with no outcome cost nothing and are simply skipped.

### What gets logged

With `log_prediction` on, each Report run appends **one prediction line** to `ComfyUI/output/preflight/feedback.jsonl` — an append-only JSONL store holding the observations, caption excerpt, caption flags, fired rule IDs, and verdicts, keyed by an 8-char `record_id`. The id is shown in the Report `summary` (`record: a3f9c2d1`) and on the `record_id` output.

### Entering an outcome

Days after posting, open the **PreFlight: Outcome** node. Its `record` dropdown lists your recent predictions, newest first, each labelled from the record itself:

```
a3f9c2d1 · 07-24 14:02 · IG:RISK TT:BLOCK X:OK · bikini/beach_pool · "summer drop 🌞 li…"
```

Pick the record, the `platform`, and the `result`, then queue. **Two caveats:**

- **The list is built when node definitions load.** New predictions appear after a **browser refresh** (or ComfyUI's Refresh button). If a record isn't in the list yet, paste its id into **`record_id_override`** — that bypasses the list entirely (the id is in the Report summary).
- Log **each platform separately** (queue once per platform). The node is cache-busted so repeated queues with identical inputs all write.

**Pick one fixed personal heuristic for `demoted` and apply it consistently.** `demoted` = clear reach suppression, an account-status flag, or FYF ineligibility. Decide *your* threshold (e.g. "≥40% below my median reach") and stick to it — inconsistent labelling poisons calibration. `clean` = normal performance; `removed` = taken down.

### Reading the calibration

```bash
python calibrate.py                 # uses ./preflight_feedback.jsonl
python calibrate.py /path/to/ComfyUI/output/preflight/feedback.jsonl
```

It prints two tables and modifies nothing:

1. **Per platform** — how often the real outcome fell inside the predicted `[best, worst]` range (consistent), **above** worst (under-predicted — the dangerous kind), or **below** best (over-predicted — you lost reach on content that was fine).
2. **Per rule** — for each fired rule ID: how many posts it fired on and the outcome distribution. This is the table that answers *"which rule is miscalibrated"* — e.g. `base.bikini` fired 30× on IG and 28 came back `clean` → the rule is too harsh.

You then edit `rules.py` and bump `ENGINE_VERSION`. Every prediction records the engine version that produced it, so cross-version comparison is free.

---

## Determinism & re-evaluating old content

Because generation is greedy and observations are stored separately with a `meta.schema_version` / `prompt_version` stamp, you can re-run `rules.judge()` over **old** observations against **new** rules without touching the model. If the schema version of a stored observation is unknown to the current engine, it **fails closed to `UNKNOWN`** rather than being silently misjudged.

---

## Licensing note

This pack is **MIT** licensed (see `LICENSE`).

Its model-management architecture is *inspired by* [ComfyUI-QwenVL](https://github.com/1038lab/ComfyUI-QwenVL), which is **GPL-3.0**. The loader here was **written independently** against `transformers` / `huggingface_hub` — no source was copied — so MIT applies. If you later copy any ComfyUI-QwenVL source verbatim into this pack, the pack must be redistributed as **GPL-3.0**. The Qwen3-VL models themselves are Apache-2.0.

---

## Limitations

Read this before trusting a verdict:

- **It is a prediction**, based on a visual observation plus a **static rule set** — not a guarantee.
- **Platform classifiers are black boxes.** The thresholds in `rules.py` are informed hypotheses, not the platforms' actual policies, which change without notice.
- **Enforcement depends on signals this node cannot see** — account history, bio, follower/link patterns, posting cadence, cluster/coordination signals, region. Two identical images on two accounts can be treated differently.
- **Thresholds need calibration against your real outcomes.** Out of the box the rules are a starting point; the feedback loop exists precisely because they will be wrong until you tune them.
- The sensor can misread. Low-confidence observations widen the verdict range on purpose.

PreFlight reports. A human decides.
