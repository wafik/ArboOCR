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

### Python (optional)

Engine facade via pybind11 — same native backends as C++. Off by default
(`ARBOOCR_BUILD_PYTHON=OFF`); enable when you have Python 3 development
headers and want `import arboocr`.

```powershell
cmake --preset windows-x64 -DARBOOCR_BUILD_PYTHON=ON
cmake --build build/windows-x64 --config Release --target _arboocr
$env:PYTHONPATH = "python"
# Windows: DLL path is auto-probed for build/windows-x64/vcpkg_installed/.../bin;
# override with $env:ARBOOCR_DLL_DIR = "...\vcpkg_installed\x64-windows\bin" if needed.
python -c "from arboocr import Engine, EngineConfig; print(EngineConfig().model_type)"
```

```python
from arboocr import Engine, EngineConfig, to_json

cfg = EngineConfig()
cfg.models_dir = "models"
cfg.use_tensorrt = False
engine = Engine(cfg)
page = engine.recognize("page.jpg")          # path
# page = engine.recognize(bgr_numpy_hxwx3)  # uint8 BGR only
for line in page.lines:
    print(line.text, line.score, [(p.x, p.y) for p in line.polygon])
print(to_json(page, pretty=True))
```

NumPy images must be **HxWx3 uint8 BGR** (OpenCV layout). No async, no
low-level Detector/Recognizer bindings, and no published wheels in v1 —
build the extension against your local `arboOCR` lib. Path overrides map to
snake_case (`rec_model_path`, etc.). See
[`docs/superpowers/specs/2026-07-22-python-bindings-design.md`](docs/superpowers/specs/2026-07-22-python-bindings-design.md).

Or skip bindings and use the bundled CLI:

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

**Defaults:** `modelType` defaults to `small` (CPU). Prefer `tiny` for
throughput, `medium` only when you have GPU/TensorRT headroom and have
measured a real accuracy win on *your* data. Detection always loads
`<ocrVersion>_det.onnx` unless you set `detModelPath` (there is no
`modelType`-selected det file).

| modelType | File size | CPU latency (SROIE warm*) | Full-page sim* |
|---|---|---|---|
| `tiny` | 4.3 MB | ~165 ms | ~93.3% |
| `small` (**default**) | ~20 MB | ~530 ms | ~94.4% |
| `medium` | 73 MB | ~2120 ms (~4× small) | ~94.8% |

<sub>*Warm Python Engine on a Windows CPU, 5 SROIE receipt images (char similarity vs box GT). Jetson / TensorRT/CUDA change absolute ms; relative ranking of sizes is the point.</sub>

We tested this on a real scanned receipt: `tiny` misread "Melawai" as
"Melavwai" and "Atas nama" as "Atasnama"; `medium` got both right with no
new errors introduced. We also tried upsizing the *detector* instead
(`PP-OCRv6_det_medium.onnx`) and got a **worse** result — 6x slower and
*more* misreads, because the larger detector's box geometry didn't suit the
recognizer it was paired with. If you want better accuracy, change
`modelType` (the recognizer), not the detector file.

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

### Accuracy defaults (read this if you upgrade)

These defaults changed in a recent accuracy cycle. **Impact is large** on
full-page text quality (SROIE receipt smoke: cold arbo small **~84% → ~94%**
char similarity, within ~0.2 pts of a TS PaddleOCR port on the same images).
If you pin config or re-implement the pipeline yourself, note every item.

| Setting | Old | New default | Why it matters |
|---------|-----|-------------|----------------|
| `modelType` | `medium` | **`small`** | Medium is ~4× slower on CPU with little sim gain on receipts. Use medium only after measuring on *your* data (GPU helps). |
| `detLimitSideLen` | `1536` | **`960`** | Longest side for det resize. 1536 over-merged / hurt full-page score on dense receipts; 960 recovered ~2 pts. Override for very high-res pages if boxes look wrong. |
| Reading order | det order only | **always sorted** (centroid y then x) | Full-page string metrics and human reading depend on order. `page.lines` is now top→bottom, left→right — **not** raw detector order. |
| CTC post | greedy only | **gap→space + fullwidth→ASCII** | Spaces injected where CTC timesteps show wide gaps (columnar fields). Fullwidth punctuation (e.g. `：`) maps to ASCII on non-CJK text. Always on in the recognizer. |
| `minimumConfidence` | (none) | **`0.5`** | Drops low-confidence lines (Paddle `drop_score`). Pure symbol lines need **0.8**. Set `0` to keep every box (legacy). |
| `splitOvermerged` | n/a | **`false`** | Optional ink-gap split of wide fused det boxes. **Off** by default — aggressive split *lost* ~1 pt on the smoke set. Enable only when det clearly fuses side-by-side fields. |
| `LinePrediction.score` | det-ish / unclear | **rec mean CTC conf** | New `detScore` holds the detector box score. JSON / Python: `score` + `det_score`. |
| CLI `--model-type` | medium | **small** | Matches library default. |

**Measured (warm Python `Engine`, CPU, 5 SROIE receipts, char sim vs box GT):**

| Engine | Size | Avg sim | Avg ms |
|--------|------|--------:|-------:|
| arbo | tiny | 93.3% | ~165 |
| arbo | **small** | **94.4%** | ~530 |
| arbo | medium | 94.8% | ~2120 |
| ppu-paddle-ocr (ref) | small | 94.6% | ~793 |

**What to watch when integrating**

1. **Line order changed.** Anything that assumed detector emission order (or
   matched boxes by index against an old dump) must use polygon geometry or
   re-sort the same way.
2. **`minimumConfidence = 0.5` drops lines.** Low-contrast logos, rules read as
   `+-`, and weak boxes disappear from `page.lines`. For audit dumps that need
   every box, set `minimumConfidence = 0`.
3. **Do not “fix accuracy” by switching to medium first.** Diagnosis on the
   smoke set: matched-line rec was already ~97%; the gap was layout / order /
   det scale, not rec capacity. Medium buys ~+0.4 pts for ~4× latency on CPU.
4. **`detLimitSideLen` is not “bigger = better”.** Try 960 before 1280/1536 on
   receipts; re-measure if your docs are posters or A3 scans.
5. **Det file is still one size.** `modelType` only selects the recognizer.
   Pairing `det_small` / `det_medium` ONNX via `detModelPath` did not help on
   the smoke set; keep the default `*_det.onnx` unless you A/B otherwise.
6. **Breaking for score consumers.** If you treated `score` as det confidence,
   switch to `detScore` / `det_score`.
7. **Opt-in only:** `useClahe`, `useAngleCls`, `splitOvermerged` — leave off
   unless the failure mode matches (faded scans / 180° / confirmed over-merge).

```cpp
// Typical production CPU defaults after the accuracy cycle (these are already
// the EngineConfig defaults — shown explicitly for upgrades):
arbo::ocr::EngineConfig cfg;
cfg.modelsDir = "models";
cfg.modelType = "small";
cfg.detLimitSideLen = 960;
cfg.minimumConfidence = 0.5f;  // 0 = keep all boxes
// cfg.splitOvermerged = true; // only if side-by-side fields fuse
// cfg.useClahe = true;        // only if det misses low-contrast text
```

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
    std::string modelType    = "small";   // recognizer size: tiny | small | medium
    float       detBoxThresh = 0.5f;
    float       detThresh    = 0.3f;
    float       detUnclipRatio = 1.6f;
    int         detLimitSideLen = 960;   // det longest side (960 beat 1536 on SROIE smoke)
    int         recBatchNum  = 6;         // crops per rec inference (raise on GPU VRAM)
    bool        useAngleCls  = false;
    bool        useCuda      = false;
    bool        useTensorrt  = false;
    bool        useFp16      = true;      // TensorRT only — FP16 kernels (default on)
    bool        useClahe     = false;     // CLAHE contrast boost before detection (faded/low-contrast docs)
    bool        splitOvermerged = false; // ink-gap split of wide fused det boxes (opt-in)
    float       minimumConfidence = 0.5f; // drop low-conf lines (0 = keep all)
    std::string trtCacheDir  = "models/trt_engines";
    std::string modelsDir    = "models";
    std::string detModelPath;   // empty = modelsDir/ocrVersion_det.onnx
    std::string clsModelPath;   // empty = modelsDir/ocrVersion_cls.onnx
    std::string recModelPath;   // empty = modelsDir/ocrVersion_rec_modelType.onnx
    std::string dictPath;       // empty = modelsDir/ocrVersion_rec_modelType_dict.txt
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

ModelPaths resolveModelPaths(const EngineConfig& cfg);
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
