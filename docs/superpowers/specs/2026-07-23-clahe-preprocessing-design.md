# CLAHE image enhancement pre-pipeline

**Date:** 2026-07-23
**Status:** Approved design
**Scope:** Optional CLAHE contrast enhancement before detection

## Context

Detection failures on low-contrast documents (e.g. faded thermal-printer
receipts) are often not a recognizer problem — DBNet never proposes a box
for text it can't see. `Engine::runPipeline()` currently feeds the raw
source image straight to `Detector::getTextBoxes()`
(`src/arboOCR/engine.cpp`) with no enhancement step. OpenCV is already a
hard dependency (`find_package(OpenCV REQUIRED)` in `CMakeLists.txt`), so
CLAHE (`cv::createCLAHE`, in `opencv2/imgproc`) is available with no new
dependency.

## Goals

1. Let callers opt into CLAHE contrast enhancement on the full image before
   detection, to recover text boxes DBNet would otherwise miss on
   low-contrast input.
2. Zero behavior change for existing callers who don't opt in.
3. Keep the new code isolated and independently testable without an ONNX
   model, matching the project's existing `ocr_utils.hpp` testing pattern.

## Non-goals

- Adaptive binarization. DBNet was trained on natural photos; binarizing
  before detection risks hurting recall rather than helping it — out of
  scope for this design.
- Per-crop CLAHE before recognition. The recognizer models weren't trained
  on CLAHE'd input; only the detector input is enhanced.
- Tunable `clipLimit`/`tileGridSize`. Fixed at OpenCV's own defaults
  (2.0 / 8x8) for v1. Add a config knob later only if a real case needs it.
- Changing `Detector`'s public API/signature.

## Design

### 1. New `applyClahe` function

New files `include/arboOCR/preprocess.hpp` / `src/arboOCR/preprocess.cpp`
(separate from `ocr_utils.hpp`, whose doc comment states its functions are
near-verbatim ports from RapidOcrOnnx — CLAHE is original arboOCR code):

```cpp
namespace arbo::ocr {

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost
/// local contrast, e.g. for faded/low-contrast scanned documents. For a
/// 3-channel image, operates on the L channel of Lab color space (preserves
/// hue/saturation) and converts back to BGR. For a 1-channel image, applies
/// CLAHE directly. Fixed clipLimit=2.0, tileGridSize=8x8 (OpenCV defaults).
/// Never throws: empty or unsupported-channel-count input is returned
/// unchanged.
cv::Mat applyClahe(const cv::Mat& src);

} // namespace arbo::ocr
```

Implementation: convert BGR→Lab, split channels, run
`cv::createCLAHE(2.0, {8,8})->apply()` on L, merge, convert back to BGR.
Guard: `src.empty()` or `src.channels()` not in `{1, 3}` returns `src`
unchanged.

### 2. `EngineConfig` flag

Add to `include/arboOCR/engine.hpp`:

```cpp
// Apply CLAHE contrast enhancement to the full image before detection.
// Off by default (matches useAngleCls/useCuda/useTensorrt pattern) — helps
// low-contrast documents (faded receipts) but adds per-image CPU cost and
// isn't universally beneficial. See preprocess.hpp::applyClahe.
bool useClahe = false;
```

### 3. Engine wiring

In `Engine::runPipeline()` (`src/arboOCR/engine.cpp`), before computing
`ScaleParam`:

```cpp
cv::Mat prepped = config_.useClahe ? applyClahe(src) : src;
ScaleParam scale = getScaleParam(prepped, config_.detLimitSideLen);
auto textBoxes = detector_.getTextBoxes(prepped, scale, ...);
...
for (auto& box : textBoxes) {
    partImages.push_back(getRotateCropImage(prepped, box.boxPoint));
}
```

Crops are taken from `prepped` (not `src`) so detector and recognizer see
consistent input — no re-deriving crops from a different source image than
the one boxes were detected on. `Detector`/`Recognizer` classes are
unchanged; they only ever see a `cv::Mat`, enhanced or not.

`prepped` is a full copy only when `useClahe` is true (CLAHE's Lab
conversion path naturally allocates new Mats); when false, `prepped` is a
cheap `cv::Mat` header copy (shared buffer, OpenCV reference-counted) — no
extra allocation in the disabled path.

### 4. CLI flag

`cli/arboocr_demo.cpp`, matching the existing `--det-model`/`--dict`-style
flags:

```cpp
("clahe", "Apply CLAHE contrast enhancement before detection", cxxopts::value<bool>()->default_value("false"))
...
cfg.useClahe = result["clahe"].as<bool>();
```

### 5. Build

`CMakeLists.txt`: add `src/arboOCR/preprocess.cpp` to the library's source
list, `tests/test_preprocess.cpp` to the test binary's source list.

### 6. README

New bullet under EngineConfig / API reference documenting `useClahe`, with
the low-contrast-receipt motivating example (mirrors the style of the
existing `useAngleCls`/model-path-override documentation).

## Error handling

- `applyClahe` never throws: empty Mat or unexpected channel count (not 1
  or 3) returns the input unchanged rather than raising.
- `Engine::runPipeline` keeps its existing never-throw contract; CLAHE
  failure modes are absorbed by `applyClahe`'s own guard, not a new
  try/catch.

## Tests

| Test | File | Expectation |
|------|------|-------------|
| Low-contrast synthetic Mat → higher stddev after `applyClahe` | `tests/test_preprocess.cpp` (new) | Contrast measurably increases |
| Empty Mat input | `tests/test_preprocess.cpp` | Returns empty Mat, no throw |
| 1-channel (grayscale) Mat input | `tests/test_preprocess.cpp` | Returns valid enhanced 1-channel Mat, no throw |
| `EngineConfig::useClahe = true` end-to-end | `tests/test_engine.cpp` (extend existing) | Pipeline runs to completion without crash/throw on the bundled sample image (no golden-output assertion — CLAHE changes pixel values, nothing to hardcode against) |
| `EngineConfig` default | `tests/test_engine.cpp` | `useClahe` defaults to `false` |

No real ONNX model required for `test_preprocess.cpp`. The `test_engine.cpp`
addition reuses whatever existing fixture/model setup that file already has.

## Alternatives considered

| Approach | Why rejected |
|----------|----------------|
| Adaptive binarization instead of/alongside CLAHE | Risks hurting DBNet recall — detector wasn't trained on binarized input |
| CLAHE inside `Detector::getTextBoxes` | Couples a class with no config-flag dependencies today to a new option; harder to use/test `Detector` standalone |
| CLAHE applied to recognizer crops too | Recognizer wasn't trained on CLAHE'd crops; risks hurting character accuracy even as detection improves |
| Tunable clipLimit/tileGridSize in v1 | YAGNI — no concrete case demands it yet; fixed OpenCV defaults are a reasonable starting point |
| Default `useClahe = true` | Changes behavior for existing callers without opt-in; CLAHE isn't universally beneficial (can amplify noise on already-good scans) |

## Success criteria

- `EngineConfig::useClahe` (default `false`) enables full-image CLAHE
  before detection with zero behavior change when left off.
- `applyClahe` is pure, independently unit-tested, and never throws.
- CLI exposes `--clahe` for manual testing against real low-contrast
  images.
- README documents the flag and when to use it.
