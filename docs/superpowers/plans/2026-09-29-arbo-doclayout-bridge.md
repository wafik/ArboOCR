# arbo-doclayout Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let arboOCR order OCR lines by sibling arbo-doclayout blocks (correct multi-column reading order), behind two opt-in gates that default off.

**Architecture:** Pure function `orderLinesByBlocks` (always compiled, unit-testable without any model) + optional CMake link to sibling library `../arbo-doclayout` + gated runtime flag in `EngineConfig`. Any failure falls back to the existing `sortLinesReadingOrder`.

**Tech Stack:** C++17, CMake ≥3.20, VS2022 + vcpkg, doctest, ONNX Runtime, OpenCV.

**Spec:** `docs/superpowers/specs/2026-09-29-arbo-doclayout-bridge-design.md` (committed on branch `feat/doclayout-bridge`).

## Global Constraints

- arboOCR repo: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR`, branch `feat/doclayout-bridge` (already exists).
- arbo-doclayout repo: `D:\kerjaan\kreasi\kimfu\cpp\arbo-doclayout`, branch `main`, clean tree.
- Windows 11, PowerShell 5.1 / Git Bash, VS 2022, vcpkg toolchain via preset `windows-x64` (needs `VCPKG_ROOT` env).
- Configure: `cmake --preset windows-x64` (from repo root). Build: `cmake --build build/windows-x64 --config Release --target <t>`. Tests exe: `build\windows-x64\Release\arboocr_tests.exe`.
- **Both gates default OFF:** `ARBOOCR_WITH_DOCLAYOUT=OFF`, `useDocLayout=false`. Default build behavior must not change in any way.
- Tests must never download models (no network in test runs).
- Do not modify: `markdown.cpp`, the body of `sortLinesReadingOrder`, detector/classifier/recognizer.
- Model cache (already populated by recon): `%USERPROFILE%\.cache\arbo-doclayout\PP-DocLayoutV2.onnx` (203 MB).
- Image for smoke: `D:\kerjaan\kreasi\kimfu\cpp\test-ocr-doclayout\data\majalah BOBO 20210303_page-0010.jpg` (path contains spaces — quote it).
- Shell stdout may be swallowed by hooks: redirect command output to a file under `build/`, then read the file.
- Conventional Commits, terse subject ≤72 chars, no AI attribution; end commit bodies with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

---

### Task 1: arbo-doclayout — gate tests/demo behind an option

**Files:**
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arbo-doclayout\CMakeLists.txt`

**Interfaces:**
- Produces: CMake option `ARBODOCLAYOUT_BUILD_TESTS` (default `ON`). Task 3 sets it `OFF` before `add_subdirectory`, so arboOCR never needs `doctest`/`cxxopts` for this subproject.

- [ ] **Step 1: Edit CMakeLists.txt — hoist find_package calls behind the option**

Replace the top block:

```cmake
find_package(doctest CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
```

with:

```cmake
option(ARBODOCLAYOUT_BUILD_TESTS "Build tests and the demo CLI" ON)

find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CURL CONFIG REQUIRED)
if(ARBODOCLAYOUT_BUILD_TESTS)
    find_package(doctest CONFIG REQUIRED)
    find_package(cxxopts CONFIG REQUIRED)
endif()
```

- [ ] **Step 2: Wrap the tests + demo block**

Wrap everything from `enable_testing()` through the final `target_link_libraries(arbo_doclayout_demo ...)` line in:

```cmake
if(ARBODOCLAYOUT_BUILD_TESTS)
    ...existing enable_testing / arbo_doclayout_tests / add_test / arbo_doclayout_demo lines unchanged...
endif()
```

Leave the `add_library(arboDocLayout ...)` block untouched and outside the `if`.

- [ ] **Step 3: Verify ON (default) still builds and passes**

```bash
cd /d/kerjaan/kreasi/kimfu/cpp/arbo-doclayout
cmake --build build/windows-x64 --config Release > build/t1_on.log 2>&1
ctest --test-dir build/windows-x64 -C Release --output-on-failure > build/t1_ctest.log 2>&1
```

Expected: build exit 0; ctest `100% tests passed` (1 suite / 13 cases).

- [ ] **Step 4: Verify OFF configures without doctest**

```bash
cmake --preset windows-x64 -DARBODOCLAYOUT_BUILD_TESTS=OFF > build/t1_off_config.log 2>&1
cmake --build build/windows-x64 --config Release --target arboDocLayout > build/t1_off_build.log 2>&1
```

Expected: both exit 0; `arbo_doclayout_tests.exe` NOT built (dir has no fresh timestamp).

Then restore the cache for normal dev: `cmake --preset windows-x64` (default ON again).

- [ ] **Step 5: Commit (arbo-doclayout repo, on `main`)**

```bash
git add CMakeLists.txt
git commit -m "build: gate tests and demo behind ARBODOCLAYOUT_BUILD_TESTS

Lets consumers add_subdirectory this project without providing
doctest/cxxopts.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: arboOCR — `orderLinesByBlocks` pure function (TDD)

**Files:**
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\include\arboOCR\ocr_utils.hpp` (after the `sortLinesReadingOrder` declaration)
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\src\arboOCR\ocr_utils.cpp` (append after `sortLinesReadingOrder`, which ends ~line 473)
- Test: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\tests\test_ocr_utils.cpp` (append at end)

**Interfaces:**
- Produces:
  - `struct arbo::ocr::LayoutBlock { std::string label; float x1, y1, x2, y2; }`
  - `bool arbo::ocr::orderLinesByBlocks(std::vector<LinePrediction>& lines, const std::vector<LayoutBlock>& blocks)` — returns false when `blocks` empty (caller falls back); never throws; `lines` untouched on false.
- Consumes: `LinePrediction` (`types.hpp`: `polygon`, `text`, `score`, `detScore`, `words`).
- Task 4 calls this from `Engine::runPipeline`.

- [ ] **Step 1: Write the failing tests — append to `tests/test_ocr_utils.cpp`**

```cpp
namespace {
LinePrediction makeLine(float x1, float y1, float x2, float y2, std::string text) {
    LinePrediction l;
    l.polygon = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
    l.text = std::move(text);
    return l;
}

std::vector<std::string> texts(const std::vector<LinePrediction>& lines) {
    std::vector<std::string> out;
    out.reserve(lines.size());
    for (const auto& l : lines) out.push_back(l.text);
    return out;
}
} // namespace

TEST_CASE("orderLinesByBlocks keeps two interleaved columns in block order") {
    std::vector<LayoutBlock> blocks = {
        {"text", 0, 0, 50, 100},
        {"text", 50, 0, 100, 100},
    };
    // naive geometric order interleaves the columns:
    std::vector<LinePrediction> lines = {
        makeLine(5, 5, 45, 12, "L1"),
        makeLine(55, 5, 95, 12, "R1"),
        makeLine(5, 20, 45, 27, "L2"),
        makeLine(55, 20, 95, 27, "R2"),
    };
    CHECK(orderLinesByBlocks(lines, blocks));
    CHECK(texts(lines) == std::vector<std::string>{"L1", "L2", "R1", "R2"});
}

TEST_CASE("orderLinesByBlocks sends orphan line to nearest block center") {
    std::vector<LayoutBlock> blocks = {
        {"text", 0, 0, 40, 30},    // top-left
        {"text", 60, 0, 100, 30},  // top-right
    };
    std::vector<LinePrediction> lines = {
        makeLine(10, 60, 35, 70, "orphan"),  // overlaps nothing; nearer left
        makeLine(5, 5, 35, 15, "L"),
        makeLine(65, 5, 95, 15, "R"),
    };
    CHECK(orderLinesByBlocks(lines, blocks));
    CHECK(texts(lines) == std::vector<std::string>{"L", "orphan", "R"});
}

TEST_CASE("orderLinesByBlocks emits header first and footer last") {
    std::vector<LayoutBlock> blocks = {
        {"footer", 0, 90, 100, 100},
        {"text", 0, 30, 100, 80},
        {"header", 0, 0, 100, 10},
    };
    std::vector<LinePrediction> lines = {
        makeLine(5, 92, 95, 98, "F"),
        makeLine(5, 35, 95, 45, "M"),
        makeLine(5, 2, 95, 8, "H"),
    };
    CHECK(orderLinesByBlocks(lines, blocks));
    CHECK(texts(lines) == std::vector<std::string>{"H", "M", "F"});
}

TEST_CASE("orderLinesByBlocks sorts lines inside one block top-to-bottom then left-to-right") {
    std::vector<LayoutBlock> blocks = {{"text", 0, 0, 100, 100}};
    std::vector<LinePrediction> lines = {
        makeLine(5, 50, 95, 60, "third"),
        makeLine(50, 10, 95, 20, "second"),
        makeLine(5, 10, 45, 20, "first"),
    };
    CHECK(orderLinesByBlocks(lines, blocks));
    CHECK(texts(lines) == std::vector<std::string>{"first", "second", "third"});
}

TEST_CASE("orderLinesByBlocks returns false on empty blocks and leaves lines untouched") {
    std::vector<LayoutBlock> blocks;
    std::vector<LinePrediction> lines = {makeLine(0, 0, 10, 10, "x")};
    CHECK_FALSE(orderLinesByBlocks(lines, blocks));
    CHECK(texts(lines) == std::vector<std::string>{"x"});
}
```

- [ ] **Step 2: Build tests — expect compile failure (function not declared)**

```bash
cmake --build build/windows-x64 --config Release --target arboocr_tests > build/t2_fail.log 2>&1
```

Expected: FAIL, `orderLinesByBlocks`: undeclared identifier / `LayoutBlock` unknown.

- [ ] **Step 3: Add declaration to `include/arboOCR/ocr_utils.hpp`**

Immediately after the `void sortLinesReadingOrder(...)` declaration block:

```cpp
/// One document layout region: label + axis-aligned box in source-image
/// coordinates. Deliberately not arboDocLayout::LayoutBox so this module —
/// and its tests — stay independent of the optional doclayout dependency.
struct LayoutBlock {
    std::string label;
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
};

/// Reorder `lines` into layout-block reading order: each line joins the
/// block covering the largest share of its area; lines touching no block
/// (orphans) join the nearest block by center distance. Blocks emit in
/// vector order (the layout model already sorts by reading order), except
/// labels starting "header" move to the front and "footer" to the back
/// (relative order kept in both groups). Lines inside a block sort by
/// top edge, then left edge. Returns false when `blocks` is empty — the
/// caller falls back to sortLinesReadingOrder — and then leaves `lines`
/// untouched. Never throws.
bool orderLinesByBlocks(std::vector<LinePrediction>& lines,
                        const std::vector<LayoutBlock>& blocks);
```

- [ ] **Step 4: Implement in `src/arboOCR/ocr_utils.cpp` — append after `sortLinesReadingOrder`**

```cpp
namespace {

struct Bounds {
    float x1, y1, x2, y2;
};

// Axis-aligned bounds of a line polygon; empty polygon collapses to zeros.
Bounds lineBounds(const LinePrediction& l) {
    float x1 = 1e9f, y1 = 1e9f, x2 = -1e9f, y2 = -1e9f;
    for (const auto& p : l.polygon) {
        x1 = std::min(x1, p.x);
        y1 = std::min(y1, p.y);
        x2 = std::max(x2, p.x);
        y2 = std::max(y2, p.y);
    }
    if (x2 < x1) return {0.f, 0.f, 0.f, 0.f};
    return {x1, y1, x2, y2};
}

// "header", "header_image", ... — PP-DocLayout header family.
bool isHeaderLabel(const std::string& label) {
    return label.rfind("header", 0) == 0;
}

} // namespace

bool orderLinesByBlocks(std::vector<LinePrediction>& lines,
                        const std::vector<LayoutBlock>& blocks) {
    if (blocks.empty() || lines.empty()) return false;

    // 1. Bucket: max (intersection / line area); orphans -> nearest center.
    std::vector<std::vector<LinePrediction>> buckets(blocks.size());
    std::vector<LinePrediction> orphans;
    for (auto& l : lines) {
        const Bounds b = lineBounds(l);
        const float area = std::max(1.f, b.x2 - b.x1) * std::max(1.f, b.y2 - b.y1);
        int best = -1;
        float bestShare = 0.f;
        for (size_t i = 0; i < blocks.size(); ++i) {
            const LayoutBlock& k = blocks[i];
            const float iw = std::min(b.x2, k.x2) - std::max(b.x1, k.x1);
            const float ih = std::min(b.y2, k.y2) - std::max(b.y1, k.y1);
            if (iw > 0.f && ih > 0.f) {
                const float share = iw * ih / area;
                if (share > bestShare) {
                    best = static_cast<int>(i);
                    bestShare = share;
                }
            }
        }
        if (best >= 0) buckets[best].push_back(std::move(l));
        else orphans.push_back(std::move(l));
    }
    for (auto& l : orphans) {
        const Bounds b = lineBounds(l);
        const float cx = (b.x1 + b.x2) * 0.5f, cy = (b.y1 + b.y2) * 0.5f;
        int nearest = 0;
        float bestD = 1e30f;
        for (size_t i = 0; i < blocks.size(); ++i) {
            const LayoutBlock& k = blocks[i];
            const float kx = (k.x1 + k.x2) * 0.5f - cx;
            const float ky = (k.y1 + k.y2) * 0.5f - cy;
            const float d = kx * kx + ky * ky;
            if (d < bestD) {
                bestD = d;
                nearest = static_cast<int>(i);
            }
        }
        buckets[nearest].push_back(std::move(l));
    }

    // 2. Block order: vector order (= model reading order), headers front,
    //    footers back; relative order preserved inside each group.
    std::vector<size_t> headers, rest, footers;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (isHeaderLabel(blocks[i].label)) headers.push_back(i);
        else if (blocks[i].label == "footer") footers.push_back(i);
        else rest.push_back(i);
    }

    // 3. Emit: lines within a block by top edge, then left edge.
    std::vector<LinePrediction> out;
    out.reserve(lines.size());
    auto appendBucket = [&](size_t i) {
        auto& bucket = buckets[i];
        std::sort(bucket.begin(), bucket.end(),
                  [](const LinePrediction& a, const LinePrediction& b) {
                      const Bounds ba = lineBounds(a), bb = lineBounds(b);
                      return ba.y1 != bb.y1 ? ba.y1 < bb.y1 : ba.x1 < bb.x1;
                  });
        for (auto& l : bucket) out.push_back(std::move(l));
    };
    for (size_t i : headers) appendBucket(i);
    for (size_t i : rest) appendBucket(i);
    for (size_t i : footers) appendBucket(i);

    lines = std::move(out);
    return true;
}
```

Requires `<algorithm>`, `<cmath>` not needed (no fabs), `<utility>` for pair-not-used — file already includes what it needs; add `<string>`/`<vector>` only if the compiler complains (ocr_utils.hpp includes types.hpp which has them).

- [ ] **Step 5: Build and run tests**

```bash
cmake --build build/windows-x64 --config Release --target arboocr_tests > build/t2_build.log 2>&1
build/windows-x64/Release/arboocr_tests.exe -tc="*orderLinesByBlocks*" > build/t2_test.log 2>&1
```

Expected: build exit 0; log shows 5 test cases, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add include/arboOCR/ocr_utils.hpp src/arboOCR/ocr_utils.cpp tests/test_ocr_utils.cpp
git commit -m "feat: add orderLinesByBlocks layout reading order

Pure function: lines -> blocks by max area share, orphans to nearest
center, headers first / footers last. No dependency on arbo-doclayout;
the bridge (Task 4) feeds it converted boxes.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: arboOCR — CMake gate `ARBOOCR_WITH_DOCLAYOUT`

**Files:**
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1's `ARBODOCLAYOUT_BUILD_TESTS` option; sibling target `arboDocLayout`.
- Produces: when `-DARBOOCR_WITH_DOCLAYOUT=ON` → compile definition `ARBOOCR_HAS_DOCLAYOUT` (PUBLIC on `arboOCR`), link to `arboDocLayout`. Task 4's `#ifdef` code compiles only then.

- [ ] **Step 1: Insert option + subdirectory before `add_library(arboOCR STATIC ...)` (line ~82, after the find_package block)**

```cmake
# Optional bridge to the sibling arbo-doclayout library (block layout for
# reading order). OFF: zero code, zero link — build identical to stock.
option(ARBOOCR_WITH_DOCLAYOUT "Link sibling arbo-doclayout for layout-block reading order" OFF)
set(ARBOOCR_DOCLAYOUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../arbo-doclayout"
    CACHE PATH "Path to arbo-doclayout sources")
if(ARBOOCR_WITH_DOCLAYOUT)
    set(ARBODOCLAYOUT_BUILD_TESTS OFF CACHE BOOL "Build arbo-doclayout tests and demo" FORCE)
    add_subdirectory("${ARBOOCR_DOCLAYOUT_DIR}" "${CMAKE_BINARY_DIR}/arbo-doclayout")
endif()
```

- [ ] **Step 2: Link + define after `target_link_libraries(arboOCR PUBLIC ${OpenCV_LIBS} CURL::libcurl arboocr_clipper)` (line ~106)**

```cmake
if(ARBOOCR_WITH_DOCLAYOUT)
    target_link_libraries(arboOCR PUBLIC arboDocLayout)
    target_compile_definitions(arboOCR PUBLIC ARBOOCR_HAS_DOCLAYOUT)
endif()
```

- [ ] **Step 3: Verify default OFF config + build unchanged**

```bash
cd /d/kerjaan/kreasi/kimfu/cpp/arboOCR
cmake --preset windows-x64 > build/t3_off_config.log 2>&1
cmake --build build/windows-x64 --config Release --target arboOCR > build/t3_off_build.log 2>&1
```

Expected: both exit 0; no `arbo-doclayout` subdir in the generated build (`grep -ri arboDocLayout build/windows-x64/CMakeCache.txt` → no match).

- [ ] **Step 4: Verify ON config + build**

```bash
cmake --preset windows-x64 -DARBOOCR_WITH_DOCLAYOUT=ON > build/t3_on_config.log 2>&1
cmake --build build/windows-x64 --config Release --target arboOCR arboocr_tests > build/t3_on_build.log 2>&1
```

Expected: exit 0; `ARBOOCR_HAS_DOCLAYOUT` in `build/windows-x64/arboOCR.vcxproj` (grep the vcxproj); arbo-doclayout subdir configured with `ARBODOCLAYOUT_BUILD_TESTS=OFF`.

- [ ] **Step 5: Run full tests in ON config**

```bash
build/windows-x64/Release/arboocr_tests.exe > build/t3_test.log 2>&1
```

Expected: all pass (Task 4's gated test does not exist yet; Task 2's tests included).

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add ARBOOCR_WITH_DOCLAYOUT gate for sibling lib

OFF (default) keeps the build identical; ON add_subdirectorys
../arbo-doclayout with its tests off and defines ARBOOCR_HAS_DOCLAYOUT.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: arboOCR — engine integration (flag, lazy engine, fallback) + demo flag

**Files:**
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\include\arboOCR\engine.hpp` (includes at top; config field after `bool useClahe = false;` ~line 70; private members after `std::string backend_ = "cpu";`)
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\src\arboOCR\engine.cpp` (helper before `runPipeline` line 187; replace line 294)
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\tests\test_engine.cpp` (`TEST_CASE("EngineConfig has sane defaults")` line 18)
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\cli\arboocr_demo.cpp` (options block ~line 55-107; cfg assembly ~line 203)

**Interfaces:**
- Consumes: Task 2 `orderLinesByBlocks`/`LayoutBlock`; Task 3 macro `ARBOOCR_HAS_DOCLAYOUT`, target `arboDocLayout` (`arbo::doclayout::Engine`, `EngineConfig{}`, `analyze(const cv::Mat&) → PageLayout{boxes[{label,x1,y1,x2,y2}],...}`).
- Produces: `EngineConfig::useDocLayout` (bool, default false, only when built ON); demo flag `--doclayout`.

- [ ] **Step 1: `engine.hpp` — gated include at top (after the existing includes, before `namespace arbo::ocr`)**

```cpp
#include <memory>
#ifdef ARBOOCR_HAS_DOCLAYOUT
#include <arboDocLayout/engine.hpp>
#endif
```

- [ ] **Step 2: `engine.hpp` — config field immediately after `bool useClahe = false;`**

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
    // Order result lines by the sibling arbo-doclayout model's blocks instead
    // of the geometric centroid sort — fixes multi-column interleaving.
    // Costs ~+440 ms/image and ~+85 MB RSS (203 MB model auto-downloaded on
    // first use). Falls back to sortLinesReadingOrder on any failure.
    // Only exists in builds configured with -DARBOOCR_WITH_DOCLAYOUT=ON.
    bool useDocLayout = false;
#endif
```

- [ ] **Step 3: `engine.hpp` — private members after `std::string backend_ = "cpu";`**

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
    /// Laid out by arbo-doclayout when useDocLayout is set. Created lazily on
    /// first use; a ctor failure (model missing/unreachable) flips
    /// docLayoutFailed_ so the engine permanently falls back to the centroid
    /// sort for the rest of the process — OCR keeps working either way.
    bool tryOrderWithDocLayout(const cv::Mat& src, std::vector<LinePrediction>& lines);
    std::unique_ptr<arbo::doclayout::Engine> docLayout_;
    bool docLayoutFailed_ = false;
#endif
```

- [ ] **Step 4: `engine.cpp` — helper definition immediately before `PagePrediction Engine::runPipeline(...)` (line 187)**

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
bool Engine::tryOrderWithDocLayout(const cv::Mat& src, std::vector<LinePrediction>& lines) {
    if (docLayoutFailed_ || src.empty() || lines.empty()) return false;
    try {
        if (!docLayout_) {
            // Empty config = PP-DocLayoutV2 defaults; model resolves from
            // ~/.cache/arbo-doclayout, downloading on first use. The ctor
            // throws if that fails — caught below.
            docLayout_ = std::make_unique<arbo::doclayout::Engine>(arbo::doclayout::EngineConfig{});
        }
        const arbo::doclayout::PageLayout page = docLayout_->analyze(src);
        std::vector<LayoutBlock> blocks;
        blocks.reserve(page.boxes.size());
        for (const auto& b : page.boxes) {
            blocks.push_back({b.label, b.x1, b.y1, b.x2, b.y2});
        }
        return orderLinesByBlocks(lines, blocks);  // false -> caller falls back
    } catch (const std::exception& ex) {
        log(LogLevel::Error, std::string("doclayout disabled for this engine: ") + ex.what());
        docLayout_.reset();
        docLayoutFailed_ = true;
        return false;
    }
}
#endif
```

(`engine.hpp` already includes `arboOCR/ocr_utils.hpp` transitively via the pipeline headers; if not, `#include "arboOCR/ocr_utils.hpp"` is already present in engine.cpp — verify with grep.)

- [ ] **Step 5: `engine.cpp` — replace line 294 `sortLinesReadingOrder(result.lines);`**

Old:

```cpp
        sortLinesReadingOrder(result.lines);
        log(LogLevel::Debug, "recognize: " + std::to_string(result.lines.size()) + " lines");
```

New:

```cpp
        bool ordered = false;
#ifdef ARBOOCR_HAS_DOCLAYOUT
        ordered = config_.useDocLayout && tryOrderWithDocLayout(src, result.lines);
#endif
        if (!ordered) {
            sortLinesReadingOrder(result.lines);
        }
        log(LogLevel::Debug, "recognize: " + std::to_string(result.lines.size()) + " lines");
```

- [ ] **Step 6: `tests/test_engine.cpp` — add to `TEST_CASE("EngineConfig has sane defaults")`, after `CHECK(cfg.useClahe == false);`**

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
    CHECK(cfg.useDocLayout == false);
#endif
```

- [ ] **Step 7: `cli/arboocr_demo.cpp` — option, after the `("clahe", ...)` option (~line 99)**

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
        ("doclayout", "Order lines by arbo-doclayout blocks (multi-column pages; +~440 ms)",
            cxxopts::value<bool>()->default_value("false"))
#endif
```

- [ ] **Step 8: `cli/arboocr_demo.cpp` — cfg assignment after `cfg.useClahe = result["clahe"].as<bool>();` (line 203)**

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
    cfg.useDocLayout = result["doclayout"].as<bool>();
#endif
```

- [ ] **Step 9: Build (ON config from Task 3) + unit tests**

```bash
cmake --build build/windows-x64 --config Release --target arboocr_tests arboocr_demo > build/t4_build.log 2>&1
build/windows-x64/Release/arboocr_tests.exe > build/t4_test.log 2>&1
```

Expected: build 0; tests all pass (now including `useDocLayout == false` check).

- [ ] **Step 10: Smoke — page-0010 with and without the flag**

```bash
build/windows-x64/Release/arboocr_demo.exe \
  --image "D:\kerjaan\kreasi\kimfu\cpp\test-ocr-doclayout\data\majalah BOBO 20210303_page-0010.jpg" \
  --json > build/t4_off.json 2> build/t4_off.err
build/windows-x64/Release/arboocr_demo.exe \
  --image "D:\kerjaan\kreasi\kimfu\cpp\test-ocr-doclayout\data\majalah BOBO 20210303_page-0010.jpg" \
  --doclayout --json > build/t4_on.json 2> build/t4_on.err
```

Expected:
- Both exit 0 (first `--doclayout` run pays the model-load, model already cached).
- OFF output: known interleaving (left/right column texts alternate).
- ON output: coherent runs — left-column lines contiguous, then right column (spot-check the first ~20 texts: no left/right alternation; header/title lines before body).
- `build/t4_on.err` may contain benign ORT `Schema error` noise (known, out of scope).

- [ ] **Step 11: Verify OFF build still compiles the old path** — reconfigure without the flag and build:

```bash
cmake --preset windows-x64 > build/t4_off_config.log 2>&1
cmake --build build/windows-x64 --config Release --target arboOCR arboocr_tests > build/t4_off_build.log 2>&1
build/windows-x64/Release/arboocr_tests.exe > build/t4_off_test.log 2>&1
```

Expected: all pass; `arboocr_demo --doclayout` now rejected as unknown option (proves the gate).

- [ ] **Step 12: Commit (re-enable ON first: `cmake --preset windows-x64 -DARBOOCR_WITH_DOCLAYOUT=ON`)**

```bash
git add include/arboOCR/engine.hpp src/arboOCR/engine.cpp tests/test_engine.cpp cli/arboocr_demo.cpp
git commit -m "feat: order lines by doclayout blocks when useDocLayout

Runtime flag (default false) gated on ARBOOCR_HAS_DOCLAYOUT; lazy
arbo-doclayout engine, ctor failure disables it for the process, any
empty result falls back to sortLinesReadingOrder. Adds --doclayout to
arboocr_demo.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: README + final verification both gates

**Files:**
- Modify: `D:\kerjaan\kreasi\kimfu\cpp\arboOCR\README.md`

**Interfaces:**
- Consumes: everything from Tasks 1-4.

- [ ] **Step 1: README — add a row to the accuracy-defaults table (the one containing the `splitOvermerged` row, ~line 454)**

```markdown
| `useDocLayout` | n/a | **`false`** | Order lines by sibling [arbo-doclayout](../arbo-doclayout) blocks — fixes multi-column reading order. Requires building with `-DARBOOCR_WITH_DOCLAYOUT=ON`. Costs ~+440 ms/image, +85 MB RSS, one-time 203 MB model download; falls back to the centroid sort on any failure. |
```

- [ ] **Step 2: README — short section after the `useClahe` example (~line 433)**

```markdown
### Layout-block reading order (optional)

The default sort is geometric (centroid row, then x) and interleaves
multi-column pages. The sibling [`arbo-doclayout`](../arbo-doclayout)
library detects layout blocks in reading order; bridge it in with two
opt-in gates, both **off by default**:

```bash
cmake --preset windows-x64 -DARBOOCR_WITH_DOCLAYOUT=ON   # gate 1: build
```

```cpp
cfg.useDocLayout = true; // gate 2: runtime — ~+440 ms/image, 203 MB model once
```

Without gate 1 the code does not exist; without gate 2 it never runs.
On any layout failure the engine falls back to the default sort.
```

- [ ] **Step 3: Final verification matrix — run all four cells**

```bash
cd /d/kerjaan/kreasi/kimfu/cpp/arboOCR
# cell 1: OFF build, tests
cmake --preset windows-x64 > build/f_off_config.log 2>&1
cmake --build build/windows-x64 --config Release --target arboocr_tests arboocr_demo > build/f_off_build.log 2>&1
build/windows-x64/Release/arboocr_tests.exe > build/f_off_test.log 2>&1
# cell 2: ON build, tests
cmake --preset windows-x64 -DARBOOCR_WITH_DOCLAYOUT=ON > build/f_on_config.log 2>&1
cmake --build build/windows-x64 --config Release --target arboocr_tests arboocr_demo > build/f_on_build.log 2>&1
build/windows-x64/Release/arboocr_tests.exe > build/f_on_test.log 2>&1
# cell 3: arbo-doclayout standalone still green (Task 1 already ran; re-run ctest)
ctest --test-dir ../arbo-doclayout/build/windows-x64 -C Release > build/f_doclayout_ctest.log 2>&1
```

Expected: all exit 0; both test logs show 0 failed; ctest 100% passed. Record final counts in the commit message.

- [ ] **Step 4: Commit + report**

```bash
git add README.md
git commit -m "docs: document useDocLayout opt-in and its costs

Co-Authored-By: Claude Code <noreply@anthropic.com>"
git log --oneline -6
```

Report to user: test counts per cell, smoke on/off verdict on page-0010, commit hashes, and any deviations from this plan.
