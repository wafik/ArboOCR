# arboOCR Python package

Thin package around the native `_arboocr` extension (pybind11).

## Build

From the repo root (after a normal C++ configure that finds OpenCV/ORT):

```powershell
cmake --preset windows-x64 -DARBOOCR_BUILD_PYTHON=ON -DPYBIND11_FINDPYTHON=ON -U "PYTHON_*"
cmake --build build/windows-x64 --config Release --target _arboocr
```

`PYBIND11_FINDPYTHON=ON` is required: without it pybind11 uses its legacy
finder and may compile against a cached `PYTHON_EXECUTABLE` rather than the
interpreter you selected. `-U "PYTHON_*"` clears such cached values.

The post-build step copies `_arboocr*.pyd` / `.so` into `python/arboocr/`.
Filenames carry the ABI tag, so builds for different interpreters coexist.

```powershell
$env:PYTHONPATH = "python"
python -m unittest discover -s python/tests -v
```

## API (v1)

- `EngineConfig` — snake_case fields matching C++ `EngineConfig`
- `Engine(cfg).recognize(path | numpy HxWx3 uint8 BGR)`
- `to_json(page|line)`, `resolve_model_paths(cfg)`, `detect_cuda()`, `detect_tensorrt()`

Not in v1: async, Detector/Classifier/Recognizer stages, PyPI wheels.
