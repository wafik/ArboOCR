# CLAHE Image Enhancement Pre-pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `EngineConfig::useClahe` flag that applies CLAHE contrast enhancement to the full source image before detection, to recover text boxes DBNet misses on low-contrast documents (e.g. faded receipts).

**Architecture:** New pure function `applyClahe(cv::Mat)` in a new `preprocess.hpp/.cpp` unit (Lab L-channel CLAHE, OpenCV defaults, never throws). `Engine::runPipeline()` calls it conditionally on `config_.useClahe` before detection, and uses the (possibly enhanced) Mat for both detection and crop extraction so detector/recognizer see consistent input. `Detector`/`Recognizer` classes are untouched.

**Tech Stack:** C++17, OpenCV (`cv::createCLAHE`, already a hard dependency), doctest, existing CMake targets (`arboOCR` lib, `arboocr_tests`, `arboocr_demo`).

**Spec:** `docs/superpowers/specs/2026-07-23-clahe-preprocessing-design.md`

---

### Task 1: `applyClahe` function + unit tests

**Files:**
- Create: `include/arboOCR/preprocess.hpp`
- Create: `src/arboOCR/preprocess.cpp`
- Create: `tests/test_preprocess.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the header**

`include/arboOCR/preprocess.hpp`:

```cpp
#pragma once
// Original arboOCR code (not a RapidOcrOnnx port) — see ocr_utils.hpp for
// the ported preprocessing helpers this file is deliberately separate from.

#include <opencv2/core.hpp>

namespace arbo::ocr {

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost
/// local contrast, e.g. for faded/low-contrast scanned documents. For a
/// 3-channel image, operates on the L channel of Lab color space (preserves
/// hue/saturation) and converts back to BGR. For a 1-channel image, applies
/// CLAHE directly. Fixed clipLimit=2.0, tileGridSize=8x8 (OpenCV defaults).
/// Never throws: empty Mat or an unsupported channel count (not 1 or 3) is
/// returned unchanged.
cv::Mat applyClahe(const cv::Mat& src);

} // namespace arbo::ocr
```

- [ ] **Step 2: Write the failing tests**

`tests/test_preprocess.cpp`:

```cpp
#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "arboOCR/preprocess.hpp"

using namespace arbo::ocr;

namespace {
double stddev(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    cv::Scalar mean, stddevScalar;
    cv::meanStdDev(gray, mean, stddevScalar);
    return stddevScalar[0];
}
} // namespace

TEST_CASE("applyClahe increases contrast on a low-contrast 3-channel image") {
    // Two mid-gray blocks with a small step between them -> low stddev.
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(120, 120, 120));
    img(cv::Rect(0, 0, 100, 50)).setTo(cv::Scalar(130, 130, 130));
    double before = stddev(img);

    cv::Mat out = applyClahe(img);
    REQUIRE_FALSE(out.empty());
    CHECK(out.channels() == 3);
    CHECK(out.rows == img.rows);
    CHECK(out.cols == img.cols);
    double after = stddev(out);
    CHECK(after > before);
}

TEST_CASE("applyClahe on an empty Mat returns empty, does not throw") {
    cv::Mat empty;
    cv::Mat out;
    CHECK_NOTHROW(out = applyClahe(empty));
    CHECK(out.empty());
}

TEST_CASE("applyClahe handles a 1-channel (grayscale) image without throwing") {
    cv::Mat gray(50, 50, CV_8UC1, cv::Scalar(100));
    gray(cv::Rect(0, 0, 50, 25)).setTo(cv::Scalar(110));
    cv::Mat out;
    CHECK_NOTHROW(out = applyClahe(gray));
    REQUIRE_FALSE(out.empty());
    CHECK(out.channels() == 1);
    CHECK(out.rows == 50);
    CHECK(out.cols == 50);
}

TEST_CASE("applyClahe on an unsupported channel count returns input unchanged") {
    cv::Mat img4(20, 20, CV_8UC4, cv::Scalar(10, 20, 30, 40));
    cv::Mat out = applyClahe(img4);
    CHECK(out.channels() == 4);
    CHECK(out.rows == 20);
    CHECK(out.cols == 20);
}
```

- [ ] **Step 3: Add new files to CMakeLists.txt, run to verify tests fail to compile/link (function not defined)**

Edit `CMakeLists.txt`:

```cmake
add_library(arboOCR STATIC
    src/arboOCR/ort_provider_utils.cpp
    src/arboOCR/ocr_utils.cpp
    src/arboOCR/preprocess.cpp
    src/arboOCR/detector.cpp
    src/arboOCR/classifier.cpp
    src/arboOCR/recognizer.cpp
    src/arboOCR/engine.cpp
    src/arboOCR/model_downloader.cpp
    src/arboOCR/types.cpp
    src/arboOCR/logging.cpp
)
```

(insert `src/arboOCR/preprocess.cpp` right after `ocr_utils.cpp`, matching existing ordering)

```cmake
add_executable(arboocr_tests
    tests/test_main.cpp
    tests/test_ocr_utils.cpp
    tests/test_preprocess.cpp
    tests/test_detector.cpp
    tests/test_classifier.cpp
    tests/test_recognizer.cpp
    tests/test_engine.cpp
    tests/test_engine_inference.cpp
    tests/test_model_downloader.cpp
    tests/test_logging.cpp
)
```

(insert `tests/test_preprocess.cpp` right after `tests/test_ocr_utils.cpp`)

Also create a stub `src/arboOCR/preprocess.cpp` with just the include and
namespace (no function body yet) so the build fails at **link** time, not
configure time:

```cpp
#include "arboOCR/preprocess.hpp"

namespace arbo::ocr {

} // namespace arbo::ocr
```

Run: `cmake --build build/windows-x64 --target arboocr_tests`
Expected: FAIL — linker error, `applyClahe` unresolved external symbol.

- [ ] **Step 4: Implement `applyClahe`**

`src/arboOCR/preprocess.cpp`:

```cpp
// Original arboOCR code (not a RapidOcrOnnx port) — see preprocess.hpp.
#include "arboOCR/preprocess.hpp"

#include <opencv2/imgproc.hpp>

namespace arbo::ocr {

cv::Mat applyClahe(const cv::Mat& src) {
    if (src.empty()) return src;
    if (src.channels() != 1 && src.channels() != 3) return src;

    auto clahe = cv::createCLAHE(/*clipLimit=*/2.0, cv::Size(8, 8));

    if (src.channels() == 1) {
        cv::Mat out;
        clahe->apply(src, out);
        return out;
    }

    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);

    cv::Mat out;
    cv::cvtColor(lab, out, cv::COLOR_Lab2BGR);
    return out;
}

} // namespace arbo::ocr
```

- [ ] **Step 5: Build and run tests, verify they pass**

Run: `cmake --build build/windows-x64 --target arboocr_tests`
Run: `build/windows-x64/arboocr_tests --test-case="applyClahe*"`
Expected: PASS, all 4 test cases green.

- [ ] **Step 6: Commit**

```bash
git add include/arboOCR/preprocess.hpp src/arboOCR/preprocess.cpp tests/test_preprocess.cpp CMakeLists.txt
git commit -m "feat: add applyClahe contrast enhancement helper"
```

---

### Task 2: Wire `useClahe` into `EngineConfig` and `Engine::runPipeline`

**Files:**
- Modify: `include/arboOCR/engine.hpp`
- Modify: `src/arboOCR/engine.cpp`
- Modify: `tests/test_engine.cpp`
- Modify: `tests/test_engine_inference.cpp`

- [ ] **Step 1: Write the failing default-value test**

In `tests/test_engine.cpp`, extend the existing `"EngineConfig has sane defaults"` test case (do not add a new one — keep all default checks together):

```cpp
TEST_CASE("EngineConfig has sane defaults") {
    EngineConfig cfg;
    CHECK(cfg.ocrVersion == "PP-OCRv6");
    CHECK(cfg.modelType == "medium"); // recognizer size; see engine.hpp doc comment
    CHECK(cfg.detBoxThresh == doctest::Approx(0.5));
    CHECK(cfg.detThresh == doctest::Approx(0.3));
    CHECK(cfg.detUnclipRatio == doctest::Approx(1.6));
    CHECK(cfg.detLimitSideLen == 1536);
    CHECK(cfg.recBatchNum == 6);
    CHECK(cfg.useAngleCls == false);
    CHECK(cfg.useFp16 == true); // TensorRT FP16 default (was always-on)
    CHECK(cfg.useClahe == false);
    CHECK(cfg.detModelPath.empty());
    CHECK(cfg.clsModelPath.empty());
    CHECK(cfg.recModelPath.empty());
    CHECK(cfg.dictPath.empty());
}
```

(only the `CHECK(cfg.useClahe == false);` line is new)

Run: `cmake --build build/windows-x64 --target arboocr_tests`
Expected: FAIL — compile error, `EngineConfig` has no member `useClahe`.

- [ ] **Step 2: Add `useClahe` to `EngineConfig`**

In `include/arboOCR/engine.hpp`, add after the `useFp16` field (before `trtCacheDir`):

```cpp
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    // full image before detection. Off by default (matches
    // useAngleCls/useCuda/useTensorrt) — helps low-contrast documents
    // (faded receipts) recover text boxes DBNet would otherwise miss, but
    // adds per-image CPU cost and isn't universally beneficial (can
    // amplify noise on already-good scans). See preprocess.hpp::applyClahe.
    bool useClahe = false;
```

Also add the include at the top of the file, alongside the other arboOCR includes:

```cpp
#include "arboOCR/preprocess.hpp"
```

- [ ] **Step 3: Build and run, verify default-value test passes**

Run: `cmake --build build/windows-x64 --target arboocr_tests`
Run: `build/windows-x64/arboocr_tests --test-case="EngineConfig has sane defaults"`
Expected: PASS

- [ ] **Step 4: Write the failing end-to-end test**

In `tests/test_engine_inference.cpp`, add a second `TEST_CASE` alongside the
existing one (same file, same skip pattern — these require local models):

```cpp
TEST_CASE("Engine with useClahe=true runs recognize without crashing"
          * doctest::skip(true)) { // un-skip once models/ is populated locally
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.modelType = "tiny";
    cfg.useAngleCls = false;
    cfg.useClahe = true;

    Engine engine(cfg);
    CHECK(engine.backend() == "cpu");

    auto result = engine.recognize("tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg");
    CHECK_FALSE(result.image.empty());
    CHECK(result.elapsedMs > 0.0f);

    std::cout << "useClahe=true — Lines detected: " << result.lines.size() << "\n";
}
```

This test is `skip(true)` like its neighbor (no local models in CI), so it
won't fail here — Step 5's manual/local verification is what actually
exercises the new code path end-to-end. That's expected: the compile step
below still catches the wiring bug even while skipped.

- [ ] **Step 5: Wire `useClahe` into `Engine::runPipeline`**

In `src/arboOCR/engine.cpp`, modify `runPipeline` (the `try` block):

```cpp
    try {
        cv::Mat prepped = config_.useClahe ? applyClahe(src) : src;
        ScaleParam scale = getScaleParam(prepped, config_.detLimitSideLen);
        auto textBoxes = detector_.getTextBoxes(prepped, scale, config_.detBoxThresh, config_.detThresh, config_.detUnclipRatio);

        std::vector<cv::Mat> partImages;
        partImages.reserve(textBoxes.size());
        for (auto& box : textBoxes) {
            partImages.push_back(getRotateCropImage(prepped, box.boxPoint));
        }
```

(only `src` → `prepped` changes in these lines, plus the new first line; the
rest of the function — classifier/recognizer calls, result assembly, catch
block — is unchanged)

Add the include at the top of `src/arboOCR/engine.cpp`, alongside
`"arboOCR/ocr_utils.hpp"`:

```cpp
#include "arboOCR/preprocess.hpp"
```

- [ ] **Step 6: Build, verify everything still compiles and non-skipped tests pass**

Run: `cmake --build build/windows-x64 --target arboocr_tests`
Run: `build/windows-x64/arboocr_tests`
Expected: PASS (all non-skipped test cases green; the two `useClahe`/model-based cases remain skipped as before — this confirms the wiring compiles and doesn't break anything already running).

If local `models/` is populated (optional, only if available on this
machine): un-skip both `test_engine_inference.cpp` cases temporarily, run
`build/windows-x64/arboocr_tests --test-case="Engine*"`, confirm the
`useClahe=true` case produces `lines.size() >= 1` on the sample receipt and
does not crash, then re-apply `* doctest::skip(true)` before committing.

- [ ] **Step 7: Commit**

```bash
git add include/arboOCR/engine.hpp src/arboOCR/engine.cpp tests/test_engine.cpp tests/test_engine_inference.cpp
git commit -m "feat: wire EngineConfig::useClahe through Engine::runPipeline"
```

---

### Task 3: CLI flag

**Files:**
- Modify: `cli/arboocr_demo.cpp`

- [ ] **Step 1: Add the `--clahe` option**

In `cli/arboocr_demo.cpp`, add to the `opts.add_options()` chain, right
after the `fp16` option and before `det-model`:

```cpp
        ("fp16", "TensorRT FP16 (default true; only used with --tensorrt)",
            cxxopts::value<bool>()->default_value("true"))
        ("clahe", "Apply CLAHE contrast enhancement before detection (helps low-contrast/faded documents)",
            cxxopts::value<bool>()->default_value("false"))
        ("det-model", "Override detector ONNX path", cxxopts::value<std::string>()->default_value(""))
```

And wire it into `cfg`, right after `cfg.useFp16 = ...`:

```cpp
    cfg.useFp16 = result["fp16"].as<bool>();
    cfg.useClahe = result["clahe"].as<bool>();
    cfg.detModelPath = result["det-model"].as<std::string>();
```

- [ ] **Step 2: Build and smoke-test the CLI**

Run: `cmake --build build/windows-x64 --target arboocr_demo`
Run: `build/windows-x64/arboocr_demo --help`
Expected: PASS — `--clahe` appears in the printed usage/options list with
its description text.

If a local sample image and models exist, also run:
`build/windows-x64/arboocr_demo --image tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg --clahe`
Expected: runs to completion, prints `Backend:`/`Lines:` output same shape
as without `--clahe`.

- [ ] **Step 3: Commit**

```bash
git add cli/arboocr_demo.cpp
git commit -m "feat(cli): add --clahe flag to arboocr_demo"
```

---

### Task 4: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add `useClahe` to the `EngineConfig` struct listing**

In `README.md`, in the `EngineConfig` struct block under `## API Reference`,
add the field right after `useFp16`:

```cpp
    bool        useFp16      = true;      // TensorRT only — FP16 kernels (default on)
    bool        useClahe     = false;     // CLAHE contrast boost before detection (faded/low-contrast docs)
    std::string trtCacheDir  = "models/trt_engines";
```

- [ ] **Step 2: Add a short usage subsection**

In `README.md`, add a new subsection right after `### Languages (PP-OCRv6)`
and before `## API Reference`:

```markdown
### Low-contrast documents (CLAHE)

Faded thermal-printer receipts and other low-contrast scans can cause the
detector to miss text boxes entirely — no amount of recognizer accuracy
fixes a box that was never detected. Set `useClahe = true` to apply CLAHE
(Contrast Limited Adaptive Histogram Equalization) to the full image before
detection:

```cpp
arbo::ocr::EngineConfig cfg;
cfg.modelsDir = "models";
cfg.useClahe = true; // boosts local contrast before the detector runs
```

Off by default — adds per-image CPU cost and isn't universally beneficial
(can amplify noise on already-good scans). Only the detector's input image
is enhanced; recognizer crops are taken from that same enhanced image for
consistency, but the enhancement itself only ever runs once per page.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document EngineConfig::useClahe"
```

---

### Task 5: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Clean rebuild of the whole project**

Run: `cmake --build build/windows-x64`
Expected: PASS — `arboOCR`, `arboocr_tests`, `arboocr_demo`, and `examples`
targets all build with no errors or new warnings.

- [ ] **Step 2: Run the full test suite**

Run: `build/windows-x64/arboocr_tests`
Expected: PASS — all non-skipped test cases green, matching the pre-change
pass count plus the new `test_preprocess.cpp` cases (4) and the extended
`EngineConfig` default check.

- [ ] **Step 3: Confirm no stray debug output or TODOs were left behind**

Run: `git diff main --stat` (or `git log --oneline main..HEAD` if already
committed per-task) to review the full set of changed files against the
plan's file list: `include/arboOCR/preprocess.hpp`,
`src/arboOCR/preprocess.cpp`, `tests/test_preprocess.cpp`,
`include/arboOCR/engine.hpp`, `src/arboOCR/engine.cpp`,
`tests/test_engine.cpp`, `tests/test_engine_inference.cpp`,
`cli/arboocr_demo.cpp`, `README.md`, `CMakeLists.txt`.
Expected: exactly this file set, no unrelated changes.
