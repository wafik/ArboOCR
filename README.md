<div align="center">

# arboOCR

**Standalone C++ OCR library — detection, orientation, and recognition, on CPU, CUDA, or TensorRT.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Jetson-lightgrey.svg)](#build)

Drop it into your own project, point it at an image, get text back.

[Quickstart](#quickstart) •
[Build](#build) •
[Models](#models) •
[API](#api-reference) •
[Benchmarks](#benchmarks) •
[Architecture](#architecture)

</div>

---

## What is this?

arboOCR runs the classic three-stage OCR pipeline — **detect** text regions,
**classify** their orientation, **recognize** the characters — over PP-OCRv6
ONNX models via ONNXRuntime. It was extracted from a larger benchmarking
harness into a single-purpose library with no baggage: no TTS, no HTTP
server, no dataset tooling, just the inference core.

- **CPU / CUDA / TensorRT**, auto-detected at runtime — construct one
  `Engine`, it picks the fastest backend available and tells you which one
  it picked. With TensorRT, **FP16 is on by default** (`EngineConfig::useFp16`)
  for edge latency; set `useFp16 = false` for FP32 engines when debugging.
- **Facade + building blocks.** Use `Engine::recognize()` for the common
  case, or drop down to `Detector`/`Classifier`/`Recognizer` directly if
  you're building a custom pipeline.
- **Batched recognition.** Text-line crops are batched (default 6 per
  inference call, matching PaddleOCR/RapidOCR; configurable via
  `EngineConfig::recBatchNum`) instead of one ONNXRuntime call per line —
  see [Benchmarks](#benchmarks) for when this actually helps.
- **In-memory + async.** `recognize(const cv::Mat&)` skips disk I/O for
  camera/API pipelines; `recognizeAsync()` returns `std::future` for
  non-blocking multi-image work.
- **Ported, not reinvented.** The detection/recognition math is a
  near-verbatim port of [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx)
  (Apache-2.0) — battle-tested logic, renamed and reorganized for a clean
  public API. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Quickstart

```cpp
#include <arboOCR/engine.hpp>
#include <iostream>

int main() {
    arbo::ocr::EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.useTensorrt = true; // auto-falls back to CUDA, then CPU
    // cfg.recBatchNum = 8; // optional: crops per rec inference (default 6)

    arbo::ocr::Engine engine(cfg);
    std::cout << "Running on: " << engine.backend() << "\n";

    // Never throws: missing/unreadable images yield empty lines + elapsedMs.
    auto page = engine.recognize("page.jpg");
    if (page.lines.empty()) {
        std::cerr << "No text found (missing image or empty page)\n";
        return 1;
    }
    for (auto& line : page.lines) {
        std::cout << line.text << " (score=" << line.score << ")\n";
        // Polygon points — draw boxes with e.g. cv::polylines
        for (auto& pt : line.polygon)
            std::cout << "  (" << pt.x << "," << pt.y << ")";
        std::cout << "\n";
    }
    // Structured export for wrappers / web backends:
    // std::cout << arbo::ocr::toJson(page, /*pretty=*/true) << "\n";
}
```

Or skip the C++ entirely and use the bundled CLI:

```bash
./arboocr_demo --image page.jpg --models-dir models
```

```
Backend: tensorrt
Image: page.jpg
Lines: 31 (402.053 ms)
  [0] "INVOICE" (score=0.845491)
  [1] "PT. Angin Sepoi" (score=0.862601)
  ...
```

More usage patterns — including driving `Detector`/`Classifier`/`Recognizer`
directly instead of the `Engine` facade — are in
[`examples/`](examples/), as small buildable programs you can run
immediately, not just read.

## Build

arboOCR ships three CMake presets. Pick the one matching your target.

| Preset | Platform | Dependency source |
|---|---|---|
| `windows-x64` | Windows, MSVC | [vcpkg](https://vcpkg.io) |
| `linux-x64` | Linux x86_64 | vcpkg |
| `jetson` | aarch64 (Jetson/embedded) | apt + vendored onnxruntime |

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

### Jetson / aarch64 (apt + self-contained onnxruntime)

vcpkg builds everything from source, which is impractical on a Jetson.
This path uses apt for OpenCV/CURL and a **fully self-contained** vendored
onnxruntime — both the C++ headers *and* the CUDA/TensorRT-enabled runtime
`.so` live under `vendor/onnxruntime/`, so arboOCR doesn't depend on any
other project's Python environment once set up.

```bash
sudo apt install -y libopencv-dev libcurl4-openssl-dev doctest-dev cxxopts-dev cmake build-essential

# 1. onnxruntime C++ headers (the pip wheel ships none). Use the latest
# release tarball whose headers are ABI-compatible with your runtime — e.g.
# v1.27.1 headers work against a 1.28.x runtime .so (the C API is stable):
mkdir -p vendor/onnxruntime && cd vendor/onnxruntime
curl -sL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-linux-aarch64-1.27.1.tgz
tar xzf ort.tgz && rm ort.tgz && mv onnxruntime-linux-aarch64-1.27.1 dist

# 2. onnxruntime runtime .so with CUDA/TensorRT support. The official aarch64
# release tarball above is CPU-only — a pip wheel (`pip install onnxruntime`)
# or JetPack's preinstalled one ships the accelerated build instead. Copy it
# in (point SRC at wherever onnxruntime is installed on your machine):
SRC=/path/to/your/onnxruntime/capi
mkdir -p lib
cp "$SRC"/libonnxruntime.so.* "$SRC"/libonnxruntime_providers_*.so lib/
ln -sf $(basename "$SRC"/libonnxruntime.so.*.*.*) lib/libonnxruntime.so.1
ln -sf $(basename "$SRC"/libonnxruntime.so.*.*.*) lib/libonnxruntime.so
cd ../..

cmake --preset jetson   # ARBOOCR_ORT_LIB_DIR defaults to vendor/onnxruntime/lib
cmake --build build/jetson -j$(nproc)
```

### Running the tests

```bash
cd build/<preset>
ctest                 # or: ./arboocr_tests
```

`Engine` auto-detects TensorRT then CUDA then CPU via
`Ort::GetAvailableProviders()`; `engine.backend()` reports what was
selected at runtime.

## Models

arboOCR needs PP-OCRv6 ONNX files in `modelsDir`:

```
models/
├── PP-OCRv6_det.onnx                text detection (single file — no size variants)
├── PP-OCRv6_cls.onnx                angle classification (only needed if useAngleCls)
├── PP-OCRv6_rec_medium.onnx         text recognition — tiny | small | medium
└── PP-OCRv6_rec_medium_dict.txt     character dict (only if not embedded in ONNX metadata)
```

**Two ways to get them:**

<details>
<summary><b>Option A — copy from an existing install</b></summary>

If you already have a Python `rapidocr` install, copy its `models/`
directory into `modelsDir`, renaming files to match the layout above.

</details>

<details>
<summary><b>Option B — download programmatically</b></summary>

```cpp
arbo::ocr::downloadOcrModels(
    "https://your-host.example/models/PP-OCRv6/", // baseUrl — you supply this
    "PP-OCRv6", "medium", "models");
```

arboOCR ships **no default download URL** — PP-OCR model hosting locations
aren't stable across mirrors, so the caller picks the source.

</details>

### Choosing a model size

`EngineConfig::modelType` (`tiny` | `small` | `medium`) selects the
**recognizer** size only — the detector has no size variants, it's always
the single `PP-OCRv6_det.onnx` file.

| modelType | File size | CPU latency* | Accuracy |
|---|---|---|---|
| `tiny` | 4.3 MB | fastest | occasional misreads on real-world noise |
| `small` | ~20 MB | middle ground | untested — worth trying if `medium` is too slow |
| `medium` (**default**) | 73 MB | ~5x slower than tiny | most accurate |

<sub>*Measured on a Jetson Nano CPU fallback, single sample image. TensorRT/CUDA narrow this gap significantly.</sub>

We tested this on a real scanned receipt: `tiny` misread "Melawai" as
"Melavwai" and "Atas nama" as "Atasnama"; `medium` got both right with no
new errors introduced. We also tried upsizing the *detector* instead
(`PP-OCRv6_det_medium.onnx`) and got a **worse** result — 6x slower and
*more* misreads, because the larger detector's box geometry didn't suit the
recognizer it was paired with. If you want better accuracy, change
`modelType` (the recognizer), not the detector file.

## API Reference

```cpp
#include <arboOCR/engine.hpp>       // Engine, EngineConfig — start here
#include <arboOCR/detector.hpp>     // Detector — text region detection
#include <arboOCR/classifier.hpp>   // Classifier — orientation (0°/180°)
#include <arboOCR/recognizer.hpp>   // Recognizer — CRNN text recognition
#include <arboOCR/logging.hpp>      // optional log callback (default: silent)
#include <arboOCR/model_downloader.hpp>
```

### `arbo::ocr::Engine`

The facade. Construct once with an `EngineConfig`, call `recognize()` per
image.

```cpp
struct EngineConfig {
    std::string ocrVersion   = "PP-OCRv6";
    std::string modelType    = "medium";  // recognizer size: tiny | small | medium
    float       detBoxThresh = 0.5f;
    float       detThresh    = 0.3f;
    float       detUnclipRatio = 1.6f;
    int         detLimitSideLen = 1536;
    int         recBatchNum  = 6;         // crops per rec inference (raise on GPU VRAM)
    bool        useAngleCls  = false;
    bool        useCuda      = false;
    bool        useTensorrt  = false;
    bool        useFp16      = true;      // TensorRT only — FP16 kernels (default on)
    std::string trtCacheDir  = "models/trt_engines";
    std::string modelsDir    = "models";
};

class Engine {
public:
    explicit Engine(const EngineConfig& config);
    std::string backend() const;                       // "tensorrt" | "cuda" | "cpu"
    PagePrediction recognize(const std::string& imagePath);
    PagePrediction recognize(const cv::Mat& image);    // in-memory (no disk I/O)
    std::future<PagePrediction> recognizeAsync(const std::string& imagePath);
    std::future<PagePrediction> recognizeAsync(const cv::Mat& image);
};
```

`recognize()` never throws — a missing/unreadable image or an inference
error degrades to an empty-lines result with `elapsedMs` still set. Check
`page.lines.empty()` for that case. Async variants are not safe for
concurrent use on the same `Engine` (one outstanding call at a time, or
one Engine per worker).

```cpp
struct LinePrediction { Polygon polygon; std::string text; float score; };
struct PagePrediction  { std::string image; std::vector<LinePrediction> lines; float elapsedMs; };

std::string toJson(const PagePrediction& page, bool pretty = false);
std::string toJson(const LinePrediction& line, bool pretty = false);
```

### Logging (optional)

The library **does not print to stdout/stderr by default**. Demos and
examples may use `std::cout`/`std::cerr` for their own UI; that is not the
core engine. For services, install a process-wide callback and route into
your stack (spdlog, glog, etc.):

```cpp
#include <arboOCR/logging.hpp>

// Demo helper:
arbo::ocr::setLogCallback(arbo::ocr::makeStderrLogger());
arbo::ocr::setMinLogLevel(arbo::ocr::LogLevel::Debug);

// Or bridge to your logger:
arbo::ocr::setLogCallback([](arbo::ocr::LogLevel level, const std::string& msg) {
    // spdlog::log(map(level), msg);
});
```

Levels: `Debug`, `Info` (default min), `Warn`, `Error`. Callbacks that throw
are swallowed so a broken sink never crashes OCR.

### Building a custom pipeline

Need more control than `Engine` gives you? The three stages are public:

```cpp
arbo::ocr::Detector detector;
detector.loadModel("models/PP-OCRv6_det.onnx", /*useCuda=*/false, /*useTensorrt=*/true);
auto boxes = detector.getTextBoxes(image, scale, boxScoreThresh, boxThresh, unclipRatio);

arbo::ocr::Recognizer recognizer;
recognizer.loadModel("models/PP-OCRv6_rec_medium.onnx");
recognizer.loadKeysFromModelMetadata(); // or loadKeysFromFile("dict.txt")
auto lines = recognizer.getTextLines(croppedImages); // batched internally
```

See [`include/arboOCR/`](include/arboOCR/) for full doc comments on each
class — every non-obvious design decision (why batching trades off the way
it does, why padding is normalized-then-padded not padded-then-normalized,
why the detector has no size variants) is documented inline where the code
lives, not just here.

### Model downloader

```cpp
DownloadResult downloadFile(const std::string& url, const std::string& destPath);
std::vector<DownloadResult> downloadOcrModels(
    const std::string& baseUrl, const std::string& ocrVersion,
    const std::string& modelType, const std::string& modelsDir);
```

## Benchmarks

All numbers measured on a **Jetson Nano**, real inference (not synthetic
timing), on a mix of document types — not just one lucky sample image.

### Across document types (tiny recognizer, CPU)

| Document | Lines | Latency |
|---|---|---|
| Table/form (bilingual, numeric) | 60 | 1.07s |
| Newspaper (multi-column) | 48 | 1.10s |
| Legal contract (dense paragraphs) | 24 | 1.01s |
| ID card (short fields) | 28 | 0.74s |
| Whiteboard menu (handwriting/marker) | 11 | 0.73s |
| Receipt | 31 | ~0.75s |

No crashes, no garbled output, across layouts ranging from dense tables to
handwritten menus.

### Batching: CPU vs. TensorRT

Recognition batches up to `recBatchNum` text-line crops per inference call
(default 6). This is a genuine trade-off, not a universal win:

| Backend | Before batching | After batching |
|---|---|---|
| CPU | ~3.9s | ~5.0s (**slower**) |
| TensorRT | ~456ms | ~340–460ms (**faster**) |

CPU has no real parallelism across the batch dimension, so padding every
crop to a shared width is wasted computation there. TensorRT/GPU backends
parallelize across the batch dimension, where batching pays off. Batching
is always on — there's currently no flag to disable it for CPU-only
deployments.

### TensorRT precision (FP16)

When `useTensorrt` is true, TensorRT builds engines with **FP16 enabled by
default** (`EngineConfig::useFp16 = true`). That is the usual edge-device
setting on Jetson / desktop GPUs: lower latency and smaller engines, with
minimal accuracy loss for OCR. Set `useFp16 = false` only if you need FP32
for debugging (expect slower first-run compile and runtime).

Changing `useFp16` (or `recBatchNum`) can invalidate cached engines under
`trtCacheDir` — clear that directory or use a separate cache path so TRT
rebuilds instead of loading a mismatched engine.

**INT8** is not exposed: it needs a representative calibration dataset and
is easy to get wrong for multilingual text. Prefer FP16 for now.

```cpp
cfg.useTensorrt = true;
cfg.useFp16 = true;   // default — keep for production edge latency
// cfg.useFp16 = false; // FP32 engines for accuracy A/B only
```

## Architecture

```
your app
   │
   ▼
┌─────────────────────────────────────────────┐
│  Engine::recognize(imagePath)                │
│                                               │
│   1. Detector::getTextBoxes()   — find lines │
│   2. Classifier::getAngles()    — 0°/180°?   │
│   3. Recognizer::getTextLines() — read text  │
│      (batched, up to 6 crops/call)           │
│                                               │
└─────────────────────────────────────────────┘
   │
   ▼
PagePrediction { lines[], elapsedMs }
```

Each stage owns its ONNXRuntime session independently and can be used on
its own. `ocr_utils.hpp` holds the shared geometry/normalization helpers
(box scoring, perspective crop, mean/norm) all three stages build on.

```
arboOCR/
├── include/arboOCR/     public headers — this is the whole API surface
├── src/arboOCR/         implementation
├── cli/                 arboocr_demo — reference CLI usage
├── examples/            basic_recognize, custom_pipeline — buildable, runnable
├── tests/               doctest suite (buildable, runnable via ctest)
├── vendor/               Clipper (vendored) + onnxruntime (user-provisioned)
└── docs/                 verification notes from real hardware testing
```

## Contributing

Issues and PRs welcome. If you're porting from or comparing against the
upstream RapidOcrOnnx/PaddleOCR algorithms, please cite the specific
upstream source/commit you're comparing against — this project tracks
those correspondences closely (see doc comments and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## License

arboOCR is licensed under the [Apache License 2.0](LICENSE).

The ported detection/recognition logic derives from
[RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx) (Apache-2.0) and
bundles [Clipper](http://www.angusj.com/clipper2/) (Boost Software License
1.0). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full
attribution.
