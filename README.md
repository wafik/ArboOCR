# arboOCR

Standalone C++ OCR library — text **detection + orientation + recognition**
using PP-OCRv6 ONNX models via ONNXRuntime (CPU / CUDA / TensorRT). Extracted
from a benchmark harness into a clean, reusable library you can drop into your
own project.

Ported from [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx)
(Apache-2.0) — see `THIRD_PARTY_NOTICES.md`.

## Quickstart

```cpp
#include <arboOCR/engine.hpp>
#include <iostream>

int main() {
    arbo::ocr::EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.useTensorrt = true; // auto-falls back to CUDA, then CPU

    arbo::ocr::Engine engine(cfg);
    auto page = engine.recognize("page.jpg");
    for (auto& line : page.lines)
        std::cout << line.text << " (" << line.score << ")\n";
}
```

Or try the bundled CLI without writing any C++:

```bash
./arboocr_demo --image page.jpg --models-dir models
```

## Build

### Windows (vcpkg)

```powershell
$env:VCPKG_ROOT = "C:\vcpkg"
cmake --preset windows-x64
cmake --build build/windows-x64 --config Release
```

### Linux x64 (vcpkg)

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset linux-x64
cmake --build build/linux-x64
```

### Jetson / aarch64 (system deps + CUDA/TensorRT onnxruntime)

vcpkg is impractical on a Jetson. Use apt packages + a fully self-contained
vendored onnxruntime (headers **and** the CUDA/TensorRT-enabled runtime
`.so`, both under `vendor/onnxruntime/`) — arboOCR doesn't depend on any
other project's Python venv once this is done:

```bash
sudo apt install -y libopencv-dev libcurl4-openssl-dev doctest-dev cxxopts-dev cmake build-essential

# 1. onnxruntime C++ headers (the pip wheel ships none). Use the latest
# release tarball whose headers are ABI-compatible with your runtime — e.g.
# v1.27.1 headers work against a 1.28.x runtime .so (the C API is stable):
mkdir -p vendor/onnxruntime && cd vendor/onnxruntime
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-linux-aarch64-1.27.1.tgz
tar xzf ort.tgz && rm ort.tgz && mv onnxruntime-linux-aarch64-1.27.1 dist

# 2. onnxruntime runtime .so with CUDA/TensorRT support. The official aarch64
# release tarball above is CPU-only — the pip wheel (`pip install onnxruntime`,
# or JetPack's preinstalled one) ships the accelerated build instead. Copy it
# in (adjust SRC to wherever onnxruntime is installed on your machine):
SRC=/path/to/your/onnxruntime/capi   # e.g. a venv's site-packages/onnxruntime/capi
mkdir -p lib
cp "$SRC"/libonnxruntime.so.* "$SRC"/libonnxruntime_providers_*.so lib/
ln -sf $(basename "$SRC"/libonnxruntime.so.*.*.*) lib/libonnxruntime.so.1
ln -sf $(basename "$SRC"/libonnxruntime.so.*.*.*) lib/libonnxruntime.so
cd ../..

cmake --preset jetson   # ARBOOCR_ORT_LIB_DIR defaults to vendor/onnxruntime/lib
cmake --build build/jetson -j$(nproc)
```

`Engine` auto-detects TensorRT then CUDA then CPU via
`Ort::GetAvailableProviders()`; `engine.backend()` reports what was selected.

## Models

arboOCR needs PP-OCRv6 ONNX files in `modelsDir`, named:

```
models/
├── PP-OCRv6_det.onnx                text detection (one file — no size variants)
├── PP-OCRv6_cls.onnx                angle classification (only if useAngleCls)
├── PP-OCRv6_rec_medium.onnx         text recognition (tiny | small | medium — see below)
└── PP-OCRv6_rec_medium_dict.txt     char dict (only if not embedded in ONNX metadata)
```

Two ways to provide them:

1. **Copy from an existing install** (e.g. a Python `rapidocr` package's
   `models/` dir) into `modelsDir`, renaming to the layout above.
2. **Download programmatically** with a base URL you control:

   ```cpp
   arbo::ocr::downloadOcrModels(
       "https://your-host.example/models/PP-OCRv6/", // baseUrl (you supply)
       "PP-OCRv6", "medium", "models");
   ```

   arboOCR ships **no** default download URL — PP-OCR model hosting URLs are
   not stable, so the caller decides the source.

### Model size trade-off (`EngineConfig::modelType`)

`modelType` (`tiny` | `small` | `medium`) selects the **recognizer** size
only — the detector is always the single `PP-OCRv6_det.onnx` file, no size
variant exists for it. Measured on the bundled sample receipt
(`tests/fixtures/test_images/`), CPU backend, Jetson Nano:

| modelType | Recognizer file size | Latency | Sample errors |
|---|---|---|---|
| `tiny` (old default) | 4.3MB | ~750ms | "Melavwai" instead of "Melawai", "Atasnama" instead of "Atas nama" |
| `medium` (**current default**) | 73MB | ~3.9s | both fixed, no new errors introduced |

`tiny` genuinely misreads characters under real-world noise (scan artifacts,
low contrast); `medium` fixed every misread we found in that pass without
introducing new ones, at roughly 5x the CPU latency. TensorRT/CUDA narrow
that latency gap significantly (the `tiny` config alone dropped from ~750ms
CPU to ~360ms TensorRT in our Jetson tests) — worth re-measuring
`medium` under TensorRT/CUDA for your own hardware if 3.9s CPU is too slow.

We also tried the analogous swap on the **detector** (`PP-OCRv6_det_medium.onnx`
in place of the tiny/default det file) and got a *worse* result: 6x slower
and *more* misreads, not fewer — the larger detector's box geometry didn't
match well with the tiny recognizer we paired it with. Net takeaway: if you
want to try a different size, change `modelType` (recognizer), not the
detector file.

Pick `tiny` if you need the fastest CPU turnaround and can tolerate
occasional misreads (or you're running under TensorRT/CUDA where the gap
matters less); pick `medium` (default) for the most accurate output; `small`
is an untested middle ground worth trying if `medium` is too slow for your
budget.

## API

- `arbo::ocr::Engine` — facade: construct with `EngineConfig`, call
  `recognize(path)` → `PagePrediction { image, lines[], elapsedMs }`.
- `arbo::ocr::Detector` / `Classifier` / `Recognizer` — the individual
  pipeline stages, exposed for custom pipelines.
- `arbo::ocr::downloadFile` / `downloadOcrModels` — optional model fetch
  helpers (libcurl).

## License

The ported OCR logic derives from RapidOcrOnnx (Apache-2.0) and bundles
Clipper (Boost Software License 1.0). See `THIRD_PARTY_NOTICES.md`.
