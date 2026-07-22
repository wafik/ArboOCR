# Multilingual v1: docs + model path overrides

**Date:** 2026-07-22  
**Status:** Approved design (Approach A)  
**Scope:** Honest PP-OCRv6 language coverage + optional model/dict path overrides

## Context

PP-OCRv6 (the default `ocrVersion` in arboOCR) is a **universal multi-language** recognizer:

- **medium / small:** one model covers **50 languages** (Simplified Chinese, Traditional Chinese, English, Japanese, and 46 Latin-script languages).
- **tiny:** ~49 languages (same family; Japanese excluded per PaddleOCR docs).

Dictionary expansion (~200 diacritics) enables single-model coverage. There is **no need** for a runtime `language` switch for the default PP-OCRv6 ONNX models arboOCR already loads.

Older PP-OCR generations shipped separate per-language recognition models (Arabic, Cyrillic, etc.). Users who still need those scripts—or a fine-tuned ONNX—must point arboOCR at custom files. Today `Engine` only builds paths as:

```
{modelsDir}/{ocrVersion}_det.onnx
{modelsDir}/{ocrVersion}_cls.onnx
{modelsDir}/{ocrVersion}_rec_{modelType}.onnx
{modelsDir}/{ocrVersion}_rec_{modelType}_dict.txt
```

There is no way to override individual paths without renaming files on disk.

## Goals

1. Document that default PP-OCRv6 is already multilingual (no false “set language = X” API).
2. Allow optional **explicit model/dict paths** for custom or non-default ONNX without breaking existing installs.
3. Keep default layout and behavior 100% backward compatible when overrides are empty.

## Non-goals

- `EngineConfig::language` (misleading for v6 universal models).
- Built-in catalog of 80 languages or default download base URLs (hosting is unstable; arboOCR intentionally ships none).
- Changing the default flat `models/` layout.
- Runtime language hot-swap on one `Engine` instance (still construct a new Engine with different paths if needed).
- INT8 / Python bindings (separate work).

## Design

### 1. `EngineConfig` path overrides

Add four optional string fields, default empty:

| Field | Meaning when empty | Meaning when set |
|--------|--------------------|------------------|
| `detModelPath` | `{modelsDir}/{ocrVersion}_det.onnx` | Load detector from this path |
| `clsModelPath` | `{modelsDir}/{ocrVersion}_cls.onnx` | Load classifier from this path (only if `useAngleCls`) |
| `recModelPath` | `{modelsDir}/{ocrVersion}_rec_{modelType}.onnx` | Load recognizer from this path |
| `dictPath` | `{modelsDir}/{ocrVersion}_rec_{modelType}_dict.txt` | Fallback keys file if ONNX metadata has no `character` field |

Resolution rule (applied at Engine construction):

```
path = override.nonempty() ? override : defaultUnderModelsDir
```

`modelType` and `ocrVersion` still matter for **default** names only; they are ignored for a component whose override path is set.

### 2. `resolveModelPaths` helper

Public pure function (header + implementation next to engine or a small `model_paths` unit):

```cpp
struct ModelPaths {
    std::string det;
    std::string cls;
    std::string rec;
    std::string dict;
};

ModelPaths resolveModelPaths(const EngineConfig& cfg);
```

- No filesystem existence checks (caller / loadModel may fail later; Engine keeps never-throw recognize semantics; construction may still throw from ORT if file missing—**existing behavior**, do not change).
- Used by `Engine` constructor so path logic is unit-testable without ONNX.

### 3. Engine constructor

Replace inline path string building with `resolveModelPaths(config)`.

Log resolved paths at Debug (or Info for backend line remains as today):

- e.g. `det=... rec=...` so embedders with a log callback can see overrides.

Dict load order unchanged:

1. `loadKeysFromModelMetadata()` on the loaded rec session  
2. else `loadKeysFromFile(resolved.dict)`  
3. warn if `keyCount() == 0`

### 4. Downloader

**No signature change** in v1.

- `downloadOcrModels(baseUrl, ocrVersion, modelType, modelsDir)` still writes default flat names.
- Custom layouts: use existing `downloadFile(url, destPath)` to place files wherever overrides point.

### 5. CLI (optional, same PR if cheap)

Add optional flags mirroring overrides:

- `--det-model`, `--cls-model`, `--rec-model`, `--dict`

Omit if implementation time is tight; not required for library correctness.

### 6. Documentation (README)

New subsection under Models (or API):

**Languages (PP-OCRv6)**

- Default ONNX is multi-language; no `language` config required for CN/EN/JP/Latin coverage.
- For scripts outside the bundled model’s training set (or custom fine-tunes), supply ONNX + set path overrides.
- Example:

```cpp
arbo::ocr::EngineConfig cfg;
cfg.modelsDir = "models";           // still used for any non-overridden component
cfg.recModelPath = "models/custom_rec.onnx";
cfg.dictPath = "models/custom_dict.txt"; // only if metadata lacks character list
```

Update API reference struct listing to include the four fields.

### 7. Tests

| Case | Expectation |
|------|-------------|
| Defaults empty | `resolveModelPaths` matches current naming with `modelsDir`/`ocrVersion`/`modelType` |
| Only `recModelPath` set | det/cls/dict still default; rec = override |
| All overrides set | all four returned as given |
| `EngineConfig` defaults | path fields empty; existing defaults unchanged |
| Logging tests | unchanged; no requirement that Engine construction runs without models |

No real-model inference required for path resolution tests.

## Error handling

- Unchanged: missing/unreadable **image** → empty `PagePrediction`.
- Missing **model file** at construct time: ORT/session load behavior unchanged (may throw today—out of scope to soften).
- Empty dict after both load methods: existing Warn log.

## Alternatives considered

| Approach | Why rejected |
|----------|----------------|
| B: `language` string that only affects logs | Misleading API (“set language” does nothing to models) |
| C: multi-lang download catalog | Wrong for v6 universal models; unstable hosting |
| Subfolder per language | Breaks existing flat `models/` installs |

## Implementation sketch (ordered)

1. Add fields to `EngineConfig` + `ModelPaths` / `resolveModelPaths`.
2. Wire `Engine` constructor + debug log of resolved paths.
3. Unit tests for resolution.
4. README languages + API docs.
5. Optional CLI flags.

## Success criteria

- Existing apps with only `modelsDir` + default filenames behave identically.
- Custom rec/dict paths load without renaming files under `models/`.
- README states PP-OCRv6 multi-lang coverage and when overrides are needed.
- Path resolution covered by unit tests without GPU/models.
