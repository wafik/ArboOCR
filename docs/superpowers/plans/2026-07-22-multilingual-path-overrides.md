# Multilingual Path Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional det/cls/rec/dict path overrides on `EngineConfig` plus honest README docs that PP-OCRv6 is already multi-language (no `language` switch API).

**Architecture:** Pure function `resolveModelPaths(EngineConfig)` maps config defaults + optional overrides to four filesystem paths. `Engine` constructor uses it instead of inline path building. Unit tests cover resolution without ONNX models. CLI exposes the same overrides as optional flags.

**Tech Stack:** C++17, CMake, doctest, cxxopts (CLI only), existing `arboOCR` static lib, optional `arbo::ocr::log` for Debug path lines.

**Spec:** `docs/superpowers/specs/2026-07-22-multilingual-path-overrides-design.md`

## Global Constraints

- No `EngineConfig::language` field.
- No default download URL / language catalog.
- Empty override strings preserve exact current default naming under `modelsDir`.
- Do not change `downloadOcrModels` signature.
- Path resolution does not check file existence.
- Engine `recognize()` never-throw contract unchanged; construct-time ORT load failures unchanged.
- Prefer TDD: failing tests first for `resolveModelPaths`.
- Windows build: `cmake --build build/windows-x64 --config Release --target arboocr_tests` then `.\build\windows-x64\Release\arboocr_tests.exe`.

## File map

| File | Role |
|------|------|
| `include/arboOCR/engine.hpp` | `EngineConfig` override fields; `ModelPaths`; `resolveModelPaths` declaration |
| `src/arboOCR/engine.cpp` | `resolveModelPaths` body; Engine ctor uses it + Debug log |
| `tests/test_engine.cpp` | Defaults + resolution unit tests |
| `cli/arboocr_demo.cpp` | Optional `--det-model` / `--cls-model` / `--rec-model` / `--dict` |
| `README.md` | Languages section + API struct fields |

No new `.cpp` translation unit required (keep resolver next to Engine).

---

### Task 1: `resolveModelPaths` + config fields (TDD)

**Files:**
- Modify: `include/arboOCR/engine.hpp`
- Modify: `src/arboOCR/engine.cpp`
- Modify: `tests/test_engine.cpp`

**Interfaces:**
- Consumes: existing `EngineConfig` (`modelsDir`, `ocrVersion`, `modelType`)
- Produces:
  - `EngineConfig::{detModelPath,clsModelPath,recModelPath,dictPath}` (`std::string`, default `""`)
  - `struct ModelPaths { std::string det, cls, rec, dict; };`
  - `ModelPaths resolveModelPaths(const EngineConfig& cfg);`

- [ ] **Step 1: Write the failing tests** in `tests/test_engine.cpp`

Append after the existing `"EngineConfig has sane defaults"` case (extend that case too):

```cpp
TEST_CASE("EngineConfig path overrides default empty") {
    EngineConfig cfg;
    CHECK(cfg.detModelPath.empty());
    CHECK(cfg.clsModelPath.empty());
    CHECK(cfg.recModelPath.empty());
    CHECK(cfg.dictPath.empty());
}

TEST_CASE("resolveModelPaths uses default flat layout under modelsDir") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.ocrVersion = "PP-OCRv6";
    cfg.modelType = "medium";
    ModelPaths p = resolveModelPaths(cfg);
    // Use path-agnostic checks: ends with expected filename; prefer
    // std::filesystem for separator safety on Windows.
    namespace fs = std::filesystem;
    CHECK(fs::path(p.det).filename() == "PP-OCRv6_det.onnx");
    CHECK(fs::path(p.cls).filename() == "PP-OCRv6_cls.onnx");
    CHECK(fs::path(p.rec).filename() == "PP-OCRv6_rec_medium.onnx");
    CHECK(fs::path(p.dict).filename() == "PP-OCRv6_rec_medium_dict.txt");
    CHECK(fs::path(p.det).parent_path() == fs::path("models"));
}

TEST_CASE("resolveModelPaths only rec override leaves others default") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.ocrVersion = "PP-OCRv6";
    cfg.modelType = "tiny";
    cfg.recModelPath = "custom/my_rec.onnx";
    ModelPaths p = resolveModelPaths(cfg);
    namespace fs = std::filesystem;
    CHECK(p.rec == "custom/my_rec.onnx");
    CHECK(fs::path(p.det).filename() == "PP-OCRv6_det.onnx");
    CHECK(fs::path(p.cls).filename() == "PP-OCRv6_cls.onnx");
    CHECK(fs::path(p.dict).filename() == "PP-OCRv6_rec_tiny_dict.txt");
}

TEST_CASE("resolveModelPaths all overrides win") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.detModelPath = "a/det.onnx";
    cfg.clsModelPath = "b/cls.onnx";
    cfg.recModelPath = "c/rec.onnx";
    cfg.dictPath = "d/dict.txt";
    ModelPaths p = resolveModelPaths(cfg);
    CHECK(p.det == "a/det.onnx");
    CHECK(p.cls == "b/cls.onnx");
    CHECK(p.rec == "c/rec.onnx");
    CHECK(p.dict == "d/dict.txt");
}
```

Also add to `"EngineConfig has sane defaults"`:

```cpp
CHECK(cfg.detModelPath.empty());
CHECK(cfg.clsModelPath.empty());
CHECK(cfg.recModelPath.empty());
CHECK(cfg.dictPath.empty());
```

(You may merge with the dedicated empty-overrides case to avoid duplication — either is fine.)

Add at top of `test_engine.cpp` if missing:

```cpp
#include <filesystem>
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cmake --build build/windows-x64 --config Release --target arboocr_tests
.\build\windows-x64\Release\arboocr_tests.exe --test-case=*resolveModelPaths*
```

Expected: compile error (`resolveModelPaths` / fields not declared) or link error — not PASS.

- [ ] **Step 3: Implement API in `include/arboOCR/engine.hpp`**

Inside `struct EngineConfig`, after `std::string modelsDir = "models";`, add:

```cpp
    // Optional absolute/relative paths. Empty = use modelsDir + default names
    // (see resolveModelPaths). Use these for custom/fine-tuned ONNX or dicts;
    // PP-OCRv6 default models are already multi-language (no language field).
    std::string detModelPath;
    std::string clsModelPath;
    std::string recModelPath;
    std::string dictPath;
```

After `struct EngineConfig { ... };`, before `detectCuda()`, add:

```cpp
struct ModelPaths {
    std::string det;
    std::string cls;
    std::string rec;
    std::string dict;
};

/// Resolve det/cls/rec/dict paths from config defaults and optional overrides.
/// Does not check that files exist. Pure / side-effect free.
ModelPaths resolveModelPaths(const EngineConfig& cfg);
```

- [ ] **Step 4: Implement `resolveModelPaths` in `src/arboOCR/engine.cpp`**

Place **before** `Engine::Engine`, in namespace `arbo::ocr` (not anonymous):

```cpp
ModelPaths resolveModelPaths(const EngineConfig& cfg) {
    fs::path modelsDir(cfg.modelsDir);
    ModelPaths out;
    out.det = !cfg.detModelPath.empty()
        ? cfg.detModelPath
        : (modelsDir / (cfg.ocrVersion + "_det.onnx")).string();
    out.cls = !cfg.clsModelPath.empty()
        ? cfg.clsModelPath
        : (modelsDir / (cfg.ocrVersion + "_cls.onnx")).string();
    out.rec = !cfg.recModelPath.empty()
        ? cfg.recModelPath
        : (modelsDir / (cfg.ocrVersion + "_rec_" + cfg.modelType + ".onnx")).string();
    out.dict = !cfg.dictPath.empty()
        ? cfg.dictPath
        : (modelsDir / (cfg.ocrVersion + "_rec_" + cfg.modelType + "_dict.txt")).string();
    return out;
}
```

- [ ] **Step 5: Wire `Engine::Engine` to use resolved paths**

Replace the four local path strings with:

```cpp
    ModelPaths paths = resolveModelPaths(config);
    log(LogLevel::Debug,
        "Model paths det=" + paths.det + " cls=" + paths.cls
        + " rec=" + paths.rec + " dict=" + paths.dict);

    // Must be set before loadModel so TensorRT profiles match runtime batch size.
    recognizer_.setRecBatchNum(config.recBatchNum);

    detector_.loadModel(paths.det, useCuda, useTensorrt, config.trtCacheDir, config.useFp16);
    if (config.useAngleCls) {
        classifier_.loadModel(paths.cls, useCuda, useTensorrt, config.trtCacheDir, config.useFp16);
    }
    recognizer_.loadModel(paths.rec, useCuda, useTensorrt, config.trtCacheDir, config.useFp16);

    if (!recognizer_.loadKeysFromModelMetadata()) {
        log(LogLevel::Debug, "Recognizer keys: model metadata missing, loading " + paths.dict);
        recognizer_.loadKeysFromFile(paths.dict);
    } else {
        log(LogLevel::Debug, "Recognizer keys: loaded from model metadata");
    }
```

Remove the old `fs::path modelsDir(...)` block that built det/cls/rec/dict strings inline.

- [ ] **Step 6: Run tests to verify they pass**

```powershell
cmake --build build/windows-x64 --config Release --target arboocr_tests
.\build\windows-x64\Release\arboocr_tests.exe
```

Expected: all non-skipped tests PASS (including new resolveModelPaths cases).

- [ ] **Step 7: Commit**

```powershell
git add include/arboOCR/engine.hpp src/arboOCR/engine.cpp tests/test_engine.cpp
git commit -m "feat: resolveModelPaths and EngineConfig model path overrides"
```

---

### Task 2: README languages + API docs

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `EngineConfig` path fields + `resolveModelPaths` from Task 1
- Produces: user-facing docs only

- [ ] **Step 1: Add Languages subsection under Models**

After the "Choosing a model size" section (or immediately after the models directory tree), insert:

```markdown
### Languages (PP-OCRv6)

Default PP-OCRv6 recognition models are **already multi-language** in a single
ONNX file (PaddleOCR: medium/small cover 50 languages including Chinese,
English, Japanese, and 46 Latin-script languages; tiny is similar without
Japanese). arboOCR has **no `language` config** — load the default models and
you get that coverage.

For scripts outside the model’s training set, or a fine-tuned ONNX, point
explicit paths at your files (empty = default names under `modelsDir`):

```cpp
arbo::ocr::EngineConfig cfg;
cfg.modelsDir = "models";
cfg.recModelPath = "models/custom_rec.onnx";
cfg.dictPath = "models/custom_dict.txt"; // only if ONNX metadata has no character list
// cfg.detModelPath / cfg.clsModelPath likewise if needed
```

Helpers: `resolveModelPaths(cfg)` returns the four resolved paths without
checking that files exist. Custom downloads: use `downloadFile(url, dest)`
into those paths; `downloadOcrModels` still writes the default flat names only.
```

- [ ] **Step 2: Update API Reference `EngineConfig` listing**

In the README API struct block, after `modelsDir`, add:

```cpp
    std::string detModelPath;   // empty = modelsDir/ocrVersion_det.onnx
    std::string clsModelPath;   // empty = modelsDir/ocrVersion_cls.onnx
    std::string recModelPath;   // empty = modelsDir/ocrVersion_rec_modelType.onnx
    std::string dictPath;       // empty = modelsDir/ocrVersion_rec_modelType_dict.txt
```

Optionally note under Engine:

```cpp
ModelPaths resolveModelPaths(const EngineConfig& cfg);
```

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "docs: PP-OCRv6 multi-language coverage and path overrides"
```

---

### Task 3: CLI path override flags

**Files:**
- Modify: `cli/arboocr_demo.cpp`

**Interfaces:**
- Consumes: `EngineConfig::{detModelPath,clsModelPath,recModelPath,dictPath}`
- Produces: CLI flags wired into config

- [ ] **Step 1: Add cxxopts options and assign to config**

In `opts.add_options()`, after existing model options, add:

```cpp
        ("det-model", "Override detector ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("cls-model", "Override classifier ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("rec-model", "Override recognizer ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("dict", "Override character dict path (fallback if no ONNX metadata)",
            cxxopts::value<std::string>()->default_value(""))
```

After existing `cfg.useFp16 = ...` assignments:

```cpp
    cfg.detModelPath = result["det-model"].as<std::string>();
    cfg.clsModelPath = result["cls-model"].as<std::string>();
    cfg.recModelPath = result["rec-model"].as<std::string>();
    cfg.dictPath = result["dict"].as<std::string>();
```

- [ ] **Step 2: Build demo**

```powershell
cmake --build build/windows-x64 --config Release --target arboocr_demo
.\build\windows-x64\Release\arboocr_demo.exe --help
```

Expected: help lists `--det-model`, `--cls-model`, `--rec-model`, `--dict`.

- [ ] **Step 3: Commit**

```powershell
git add cli/arboocr_demo.cpp
git commit -m "feat(cli): model path override flags for arboocr_demo"
```

---

### Task 4: Final verification

**Files:** none (verify only)

- [ ] **Step 1: Full test suite**

```powershell
cmake --build build/windows-x64 --config Release --target arboocr_tests arboocr_demo
.\build\windows-x64\Release\arboocr_tests.exe
```

Expected: all tests PASS (skipped inference test still skipped if models absent).

- [ ] **Step 2: Grep sanity — no language API**

```powershell
rg "language" include/arboOCR src/arboOCR --glob "*.{hpp,cpp}"
```

Expected: no `EngineConfig::language` or similar API. Mentions only in comments/docs are fine if any.

- [ ] **Step 3: Confirm success criteria from spec**

- Empty overrides → same default filenames as before.
- Overrides load without renaming under `models/`.
- README documents multi-lang v6 + overrides.
- Path tests run without GPU/models.

If any fail, fix before considering done. No extra commit if already clean.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Four optional path fields on `EngineConfig` | Task 1 |
| `ModelPaths` + `resolveModelPaths` | Task 1 |
| Engine ctor uses resolver + Debug log | Task 1 |
| Unit tests: default / partial / full override | Task 1 |
| No downloader signature change | (explicit non-work) |
| README languages + API fields | Task 2 |
| Optional CLI flags | Task 3 |
| No `language` field | Task 4 grep |

## Placeholder / consistency self-review

- No TBD/TODO left in steps.
- Symbol names consistent: `detModelPath`, `clsModelPath`, `recModelPath`, `dictPath`, `ModelPaths`, `resolveModelPaths`.
- Default filenames match current engine.cpp: `{ocrVersion}_det.onnx`, `_cls.onnx`, `_rec_{modelType}.onnx`, `_rec_{modelType}_dict.txt`.
