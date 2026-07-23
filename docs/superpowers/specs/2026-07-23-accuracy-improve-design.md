# arboOCR accuracy & latency honesty improve

**Date:** 2026-07-23  
**Status:** Approved design (pending user review of this file)  
**Scope:** Measure-then-fix accuracy gap vs ppu-paddle-ocr; score API honesty; CPU-safe defaults  
**Related:** SROIE compare in `cpp/compare/` (`RESULTS.md`, `compare_ocr.ts`); CLAHE spec is separate (`2026-07-23-clahe-preprocessing-design.md`)

## Context

On a 5-image SROIE2019 smoke set (CPU, PP-OCRv6 tiny/small/medium):

| Engine | Size | Avg ms | Avg char-sim vs box GT |
|--------|------|-------:|-----------------------:|
| ppu | small | 870 | **94.6%** |
| arbo | small | 785 | **84.1%** |
| arbo | medium | 3535 | **84.1%** |

Findings that drive this design:

1. **Accuracy is flat across arbo sizes** (~83–84% sim). Scaling rec to medium does not buy accuracy → bottleneck is unlikely “need a bigger rec model.”
2. **Medium is very slow on CPU** (~60–80 ms/line) because det_medium + rec_medium are large and cost scales with line count. Not a mystery bug; wrong default for CPU.
3. **`LinePrediction.score` is the detection box score**, not recognition confidence (`engine.cpp` assigns `textBoxes[i].score` while `RawTextLine` has `charScores`). CLI “score=” misleads tuning and comparisons.
4. Compare harness **spawned `arboocr_demo` per image**, so wall latency overstates arbo vs long-lived ppu service.
5. Det default in library is `*_det.onnx` (no size); the compare runner paired `det_{size}` for “fairness,” which made medium even more expensive.

Baseline reference: `cpp/compare/RESULTS.md`.

## Goals

1. **Diagnose** whether the ~10 pt sim gap vs ppu is mostly **det**, **rec**, or **mixed** (evidence from same-crop / IoU / line metrics).
2. **Fix** only what the diagnosis proves (plus known score wiring bug).
3. **Latency honesty:** default away from medium on CPU; document det-size policy; report **warm** path numbers.
4. **Re-bench** against ppu on the same 5 SROIE images and update `RESULTS.md`.

## Non-goals

- GPU / TensorRT performance campaign as the primary deliverable.
- Official SROIE entity F1 as a required gate (optional later).
- Merging CLAHE by default (separate approved design; enable only if diagnosis shows low-contrast misses).
- Forcing bit-identical ONNX/ORT weights with ppu.
- Rewriting detector math from scratch / full RapidOCR re-port.
- **Integrating `arbo-doclayout` (PP-DocLayoutV2) into the arboOCR pipeline in this cycle.**  
  Sibling repo at `cpp/arbo-doclayout`: region/layout labels (text/table/figure/…), not line OCR.  
  Its own README says v1 does **not** link arboOCR (“merge later”).  
  The measured ~10 pt sim gap vs ppu is on **classic det→rec** receipts; ppu’s winning numbers also come from that stack, not DocLayout.  
  DocLayout is the wrong first lever for “medium slow / sim flat across sizes.”  
  **Revisit later** if diagnosis shows reading-order / multi-column / table-region failures, or when building a product pipeline that crops regions before OCR.

## Approach

**Measure-first toolkit + surgical library fixes** (not guess-and-patch thresholds, not a rewrite).

Work lives in:

- **Main tree** `cpp/arboOCR` for library/CLI/default changes.
- **`cpp/compare`** for harness, diagnosis scripts, and RESULTS updates.

## Design

### 1. Score API honesty

**Problem:** `LinePrediction.score` documents “detection score” in a comment sense but is what demos print as the line quality signal; callers and humans read it as OCR confidence.

**Change:**

- Set `LinePrediction.score` to a **recognition confidence** derived from `RawTextLine::charScores` (mean of non-blank char scores, or the same aggregate RapidOCR-style path already used internally if one exists — pick one definition and document it in `types.hpp`).
- Add `LinePrediction.detScore` for the detector box score so both remain available.
- Update `toJson`, Python bindings, CLI printout, and examples to show both when useful (`score` = rec, `detScore` = det).
- **Compatibility:** treat this as an intentional semantic fix. No known stable external consumer of “score = det” is assumed; changelog note required.

**Test:** unit or engine-level assertion that when rec returns known `charScores`, `LinePrediction.score` reflects them (not the det box score). Prefer pure wiring test if fixtures allow; else extend an existing recognizer/engine test.

### 2. Defaults and det-size policy

| Item | Change |
|------|--------|
| `EngineConfig::modelType` default | `"medium"` → **`"small"`** |
| Det path resolution | **Unchanged:** default file is `<ocrVersion>_det.onnx` (no size suffix). Size-specific det files only via `detModelPath`. |
| Docs / README | One short subsection: CPU default is small; medium is for accuracy-at-cost (prefer GPU); det is not selected by `modelType`. |
| CLI default | Follow library default (`small`). |

Rationale: measured medium CPU cost is high and sim did not improve; small matches “honest default.”

### 3. Compare / diagnosis harness (`cpp/compare`)

Extend or add scripts next to `compare_ocr.ts` (TypeScript/Bun, same stack as existing compare). Prefer reusing paths/env already used for arbo models and SROIE.

| Deliverable | Responsibility |
|-------------|----------------|
| **Warm fair compare** | Long-lived ppu `PaddleOcrService` per size; long-lived arbo via **Python `Engine`** (`PYTHONPATH=python`, existing bindings). Fallback if Python/DLL fails: small C++ multi-image loop is optional follow-up, not a blocker if Python works on this machine. Report warm ms + sim for **tiny** and **small** (medium optional). |
| **`diag_det_rec`** | Per image: arbo det → boxes/crops; arbo rec on those crops; line count vs GT; IoU match boxes to SROIE `test/box` polygons when feasible; rec-only / full-page sim. Verdict file: **det-bound / rec-bound / mixed** with numbers. |
| **Same-crop vs ppu** | Best effort: if ppu can score crops (`ArrayBuffer` of crop PNGs or full-page only), compare; if not practical, rely on IoU-matched line text vs GT and ppu full-page baseline. Do not block Phase 4 on perfect ppu-crop parity. |
| **`sweep_det`** | **Only if** diagnosis is det-bound or mixed. Small grid over `detThresh`, `detBoxThresh`, `detUnclipRatio`, `detLimitSideLen`. Cap grid size; keep 2–3 images as informal holdout if tuning on the 5. |

Harness must **not** silently force `det_medium` unless that pair is the explicit experiment. Default arbo measure path: library defaults (small rec + default det file), with optional overrides documented.

### 4. Proven pipeline fix (Phase 4 only)

After diagnosis writes a verdict:

| Verdict | Allowed fix class (minimal) |
|---------|-----------------------------|
| **Det-bound** | Adjust default det thresholds and/or document recommended overrides; optional CLAHE opt-in only if low-contrast misses dominate |
| **Rec-bound** | Rec preprocess alignment, dict/CTC/space handling, batch edge cases |
| **Mixed** | Smallest pair of fixes that each address a measured failure mode — still prefer one PR-sized change set |

No “turn on medium by default” as an accuracy fix. No large refactors without measurement.

### 5. Re-bench and docs

- Re-run warm compare on the same 5 images.
- Update `cpp/compare/RESULTS.md` with before/after and diagnosis summary.
- Note remaining gap vs ppu and open follow-ups (entity F1, GPU, larger sample).

## Phased plan

| Phase | Work | Exit criteria |
|-------|------|----------------|
| **1. Instrumentation** | Rec `score` + `detScore`; default `modelType=small`; README det policy | `arboocr_tests` green; demo/CLI shows rec score |
| **2. Warm fair compare** | Long-lived ppu + arbo; tiny/small table | Warm ms + sim table without per-image process spawn for arbo |
| **3. Diagnosis** | `diag_det_rec` (+ IoU / line metrics) | Written verdict: det / rec / mixed with evidence |
| **4. One proven fix** | Minimal pipeline change from Phase 3 | arbo small sim **> 84.1%** baseline on the 5-image set |
| **5. Re-bench** | RESULTS.md + residual gap notes | Full package checklist below satisfied |

## Success criteria (cycle complete)

1. Rec confidence exposed correctly (`score`); det available as `detScore`.
2. Default `modelType` is **small**; docs match CPU reality; det not implied by `modelType`.
3. Warm latency numbers exist for arbo vs ppu (at least tiny + small).
4. Written diagnosis names primary bottleneck with evidence.
5. At least one pipeline fix landed **because** of that evidence (not only docs).
6. arbo **small** mean sim on the 5-image set **improves** over baseline **84.1%**.  
   - Stretch: ≥ **89%**.  
   - Stretch-2: within ~2 pts of ppu small (~94.6%).  
   Stretch targets are aspirational; criterion 6 is “strictly better with evidence-based fix.”
7. Claiming the accuracy win does **not** require medium.

**Hard fail:** sim unchanged and only docs/defaults shipped without a measured pipeline fix attempt after diagnosis (defaults-only is incomplete for this “full package” request).

## Testing

- All existing `arboocr_tests` remain green.
- New/updated test for score wiring (rec vs det).
- Harness scripts are the acceptance checks for Phases 2–5 (runnable, no heavy test framework).
- Prefer one small self-check or documented command block in RESULTS / script header.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Python `Engine` fails on Windows (DLL / ORT schema noise) | PATH + `ARBOOCR_DLL_DIR`; if blocked, add minimal C++ multi-image warm CLI as follow-up |
| Same-crop ppu rec awkward | IoU + GT line metrics; full-page ppu baseline remains valid |
| Det sweep overfits 5 images | Small grid; holdout images; prefer generalizable default changes |
| Scope creep (CLAHE, GPU, rewrite) | Non-goals; Phase 4 limited to proven class |
| Semantic change of `score` breaks a caller | Changelog; `detScore` preserves det signal |

## File map (expected touch set)

**arboOCR**

- `include/arboOCR/types.hpp` — `detScore`; document `score`
- `src/arboOCR/engine.cpp` — wire rec + det scores
- `src/arboOCR/types.cpp` — JSON fields
- `include/arboOCR/engine.hpp` — default `modelType`
- `cli/arboocr_demo.cpp` — print fields
- `python/bindings/module.cpp` — expose `det_score` if needed
- `README.md` — defaults + det policy
- `tests/*` — score wiring
- This spec under `docs/superpowers/specs/`

**compare**

- `compare_ocr.ts` and/or new `bench_warm.ts`, `diag_det_rec.ts`, optional `sweep_det.ts`
- `RESULTS.md` — before/after

Exact script names may collapse into fewer files if that stays clearer (ponytail: fewest files that work).

## Implementation order

After this spec is user-approved:

1. Invoke **writing-plans** → step plan under `docs/superpowers/plans/`.
2. Implement Phase 1 → 5 in order; do not skip diagnosis before pipeline “improvements.”

## Open decisions (resolved in brainstorming)

| Topic | Decision |
|-------|----------|
| Success shape | Full package (accuracy + latency defaults + scores) |
| Workspace | Main tree arboOCR + compare harness |
| Sequence | Measure then fix (not defaults-only first) |
| Approach | Measure-first toolkit + surgical fixes |
| Score API | `score` = rec; add `detScore` |
| Warm bench | Prefer Python Engine; C++ multi-image optional fallback |
| arbo-doclayout | Out of scope this cycle; optional later if layout/reading-order is the failure mode |
