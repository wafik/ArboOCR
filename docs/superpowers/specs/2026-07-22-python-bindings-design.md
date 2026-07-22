# Python bindings v1 (Engine facade)

**Date:** 2026-07-22  
**Status:** Approved design  
**Scope:** pybind11 bindings for `Engine` / `EngineConfig` / prediction types only

## Goals

1. Let Python users construct an OCR engine and run recognition with the same
   native backends (CPU / CUDA / TensorRT) as C++.
2. Accept image **path** (`str`/`pathlib.Path`) or **NumPy** `uint8` HxWx3 BGR.
3. Return structured line results (text, score, polygon) and optional JSON.
4. Integrate with existing CMake via opt-in flag; no forced Python dependency
   for pure C++ builds.

## Non-goals (v1)

- PyPI multi-platform wheel CI / scikit-build packaging for release
- Binding `Detector` / `Classifier` / `Recognizer` low-level APIs
- `recognizeAsync` / `std::future` (callers use threads in Python if needed)
- `downloadOcrModels` / logging callback from Python
- Automatic RGB→BGR (caller supplies BGR, matching OpenCV `imread`)
- Shipping ONNX models inside the wheel

## Approaches considered

| Approach | Notes |
|----------|--------|
| **A (chosen)** | CMake `ARBOOCR_BUILD_PYTHON=ON` + pybind11 + thin `python/arboocr` package |
| B | scikit-build-core owns full C++ build — better for future wheels, heavier now |
| C | Hand-written `setup.py` with hard-coded paths — fragile |

## Architecture

```
python/arboocr/
  __init__.py          # re-exports Engine, EngineConfig, helpers from extension
  _arboocr*.pyd/.so    # pybind11 module (built by CMake, copied next to package)

python/bindings/
  module.cpp           # PYBIND11_MODULE(_arboocr, m)

CMake:
  option(ARBOOCR_BUILD_PYTHON OFF)
  when ON: find/FetchContent pybind11, Python3, build module linked to arboOCR
```

### Exposed API (Python)

```python
from arboocr import Engine, EngineConfig, to_json

cfg = EngineConfig()
cfg.models_dir = "models"       # snake_case in Python
cfg.model_type = "medium"
cfg.use_tensorrt = False
cfg.use_fp16 = True
# path overrides: det_model_path, cls_model_path, rec_model_path, dict_path
# thresholds: det_box_thresh, det_thresh, det_unclip_ratio, det_limit_side_len
# rec_batch_num, use_angle_cls, use_cuda, trt_cache_dir, ocr_version

engine = Engine(cfg)
print(engine.backend())         # "cpu" | "cuda" | "tensorrt"

page = engine.recognize("page.jpg")
# or: engine.recognize(numpy_array)  # HxWx3 uint8 BGR

for line in page.lines:
    print(line.text, line.score, line.polygon)  # polygon: list of (x, y) or Point2f

print(to_json(page, pretty=True))
```

### C++ ↔ Python mapping

| C++ | Python |
|-----|--------|
| `EngineConfig` fields (camelCase) | snake_case properties (read/write) |
| `Engine::recognize(string)` | `Engine.recognize(str \| Path)` |
| `Engine::recognize(cv::Mat)` | `Engine.recognize(numpy.ndarray)` via buffer protocol |
| `PagePrediction` | class with `image`, `lines`, `elapsed_ms` |
| `LinePrediction` | `text`, `score`, `polygon` |
| `Point2f` | `x`, `y` (or tuple-friendly) |
| `toJson` | `to_json(page, pretty=False)` |
| `resolveModelPaths` | optional `resolve_model_paths(cfg) → dict` |

### NumPy rules

- Require `ndim == 3`, `shape[2] == 3`, `dtype == uint8`.
- Interpret as **BGR** contiguous (or make contiguous clone).
- Wrong dtype/shape → raise `ValueError` (clear message). Empty result from
  C++ still maps to empty `lines` (C++ never-throw recognize contract).

### Engine construction failures

- If C++ constructor throws (missing ONNX), translate to Python exception
  (`RuntimeError` with what()).

### Threading

- Document: one outstanding `recognize` per `Engine` (same as C++).
- Module not required to be free-threaded; default GIL held during
  `recognize` is acceptable for v1 (GIL release optional stretch).

## Build

1. `vcpkg.json`: add `pybind11` only if using vcpkg path; **or**
   `FetchContent` pybind11 when `ARBOOCR_BUILD_PYTHON=ON` so C++-only
   consumers never pull it. Prefer **FetchContent** to keep default
   `vcpkg install` unchanged.
2. `find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)`
3. `pybind11_add_module(_arboocr python/bindings/module.cpp)`
4. Link `arboOCR`; include dirs from target.
5. Post-build: copy module into `python/arboocr/` for editable use.

### Developer workflow (docs)

```text
cmake --preset windows-x64 -DARBOOCR_BUILD_PYTHON=ON
cmake --build build/windows-x64 --config Release --target _arboocr
# set PYTHONPATH=python or pip install -e python/
python -c "from arboocr import Engine, EngineConfig; ..."
```

Optional minimal `python/pyproject.toml` for `pip install -e .` that only
packages pure Python + expects prebuilt extension (document clearly).

## Tests

- C++ suite unchanged when `ARBOOCR_BUILD_PYTHON=OFF`.
- Python smoke (skip if no models / no extension built):
  - import module
  - `EngineConfig` defaults / snake_case set
  - `resolve_model_paths` filenames
  - `recognize` missing image → empty lines (if Engine can construct)
- Prefer a small `python/tests/test_smoke.py` run manually or via CTest
  `add_test` when Python build is on.

## README

Short “Python” subsection: enable flag, build target, import example,
BGR note, non-goals (no async / no wheels yet).

## Success criteria

- `ARBOOCR_BUILD_PYTHON=OFF` (default): identical C++ build as today.
- With flag ON: importable `arboocr`, path + numpy recognize work against
  existing models when present.
- No `language` API; path overrides available via snake_case fields.

## Out of scope follow-ups

- Wheel CI, manylinux, CUDA wheel variants
- Full pipeline stage bindings
- RGB convenience flag
- WASM (see separate parked note)
