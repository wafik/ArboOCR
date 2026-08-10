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

Missing model files are downloaded on first run (see [Models](#models)).
The flags and environment variables that control that:

| Flag / env var | Effect |
|---|---|
| `--download-models` | Prefetch the models and exit — prints `ok` / `skipped` / `absent` / `MISSING` per file; exit 0 when det+rec are present, else 2 |
| `--no-download` | Never touch the network; use only what's already on disk |
| `--models-url <url>` | Fetch from your own mirror instead of the pinned default release |
| `ARBOOCR_OFFLINE=1` | Same as `--no-download`, process-wide |
| `ARBOOCR_MODELS_URL` | Default base URL override |
| `ARBOOCR_CACHE_DIR` | Override the per-user model cache directory |

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

arboOCR runs on PP-OCRv6 ONNX files. By default it fetches them for you on
first use; if you'd rather supply your own, this is the layout it expects in
`modelsDir`:

```
models/
├── PP-OCRv6_det.onnx                text detection (single file — no size variants)
├── PP-OCRv6_cls.onnx                angle classification (only needed if useAngleCls)
├── PP-OCRv6_rec_medium.onnx         text recognition — tiny | small | medium
└── PP-OCRv6_rec_medium_dict.txt     character dict (only if not embedded in ONNX metadata)
```

**Three ways to get them:**

<details>
<summary><b>Option A — do nothing (default: auto-download)</b></summary>

Construct an `Engine` with an empty `modelsDir` and arboOCR fetches what it
needs from its own pinned release, verifies each file against a SHA-256
compiled into the binary, and caches it per-user:

| Platform | Cache location |
|---|---|
| Windows | `%LOCALAPPDATA%\arboOCR\models\models-v1` |
| macOS | `~/Library/Caches/arboOCR/models/models-v1` |
| Linux | `$XDG_CACHE_HOME/arboOCR/models/models-v1`, else `~/.cache/arboOCR/models/models-v1` |

The cache path carries the release tag, so a future `models-v2` can never
reuse a `models-v1` file. Writes are atomic: the download lands on a sibling
`.<pid>.tmp` path, gets hashed, and is only renamed onto the destination
once the hash matches — a killed process or a truncated transfer can never
leave a half-written model behind, and a cached file that no longer matches
its pinned hash is re-fetched rather than trusted.

Prefetch and exit — for CI, Docker image builds, or staging an air-gapped
box:

```bash
./arboocr_demo --download-models --models-dir models
```

Turn it off with `cfg.autoDownload = false`, `--no-download`, or
`ARBOOCR_OFFLINE=1`.

</details>

<details>
<summary><b>Option B — copy from an existing install</b></summary>

If you already have a Python `rapidocr` install, copy its `models/`
directory into `modelsDir`, renaming files to match the layout above. A
populated `modelsDir` short-circuits the download — no network traffic.

</details>

<details>
<summary><b>Option C — your own mirror, or fine-tuned weights</b></summary>

```cpp
// Point the auto-download at your own host:
cfg.modelsBaseUrl = "https://your-host.example/models/PP-OCRv6/";

// Or fetch explicitly, without constructing an Engine ("" baseUrl = the default release):
arbo::ocr::downloadOcrModels(
    "https://your-host.example/models/PP-OCRv6/",
    "PP-OCRv6", "medium", "models");
```

This section used to say arboOCR ships **no default download URL**, because
"PP-OCR model hosting locations aren't stable across mirrors, so the caller
picks the source." That is no longer true, and the reason it changed is
worth stating rather than quietly deleting.

The objection was really three objections: a hardcoded URL rots, you don't
control the integrity of what comes back, and you don't control its
provenance. A default download is only defensible once all three are
answered. They now are — the URL resolves to an immutable pinned release tag
(`models-v1`), every file is checked against a SHA-256 baked into the
binary, and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records where
the weights came from.

The escape hatch is unchanged, and it still wins. Precedence, per file:

1. An explicitly set `detModelPath` / `clsModelPath` / `recModelPath` /
   `dictPath` is returned as-is and is **never** substituted by a download —
   a fine-tuned model is never silently swapped for a stock one.
2. An existing non-empty file under `modelsDir`.
3. Only then, the download.

`downloadOcrModels` looks up the pinned hash by file name, so a custom
mirror serving the stock names still gets integrity checking.

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
checking that files exist — it is pure and never touches the network.
`ensureOcrModels(cfg)` is what `Engine`'s constructor actually calls: the
same four paths, but anything missing is downloaded and hash-verified first,
and explicitly set paths are always returned untouched.

```cpp
ModelPaths ensureOcrModels(const EngineConfig& cfg);
```

`modelDownloadsAllowed(cfg)` is the single answer to "may this config touch
the network?" — `cfg.autoDownload` **and** `ARBOOCR_OFFLINE` unset (or `0`).
`ensureOcrModels` gates on it; call it rather than reading `cfg.autoDownload`
if you report on downloading in your own error messages.

Custom downloads: use `downloadFile(url, dest, expectedSha256)` into those
paths; `downloadOcrModels` still writes the default flat names only.

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
    bool        autoDownload = true; // fetch missing models on construction (ARBOOCR_OFFLINE=1 disables)
    std::string modelsBaseUrl;  // empty = defaultModelsBaseUrl() (pinned release)
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

ModelPaths resolveModelPaths(const EngineConfig& cfg);   // pure — no filesystem, no network
ModelPaths ensureOcrModels(const EngineConfig& cfg);     // resolves, then downloads what's missing
bool modelDownloadsAllowed(const EngineConfig& cfg);     // cfg.autoDownload AND not ARBOOCR_OFFLINE
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
const char* defaultModelsTag();       // "models-v1" — pinned, immutable
std::string defaultModelsBaseUrl();   // the release assets that tag points at
std::string defaultModelsCacheDir();  // per-user, tag-scoped (ARBOOCR_CACHE_DIR overrides)

std::string sha256File(const std::string& path);
std::string knownSha256(const std::string& fileName); // "" if that name isn't pinned

DownloadResult downloadFile(const std::string& url, const std::string& destPath,
                            const std::string& expectedSha256 = "");
std::vector<DownloadResult> downloadOcrModels(
    const std::string& baseUrl, const std::string& ocrVersion,
    const std::string& modelType, const std::string& modelsDir);
```

With an `expectedSha256`, `downloadFile` writes to a sibling `.<pid>.tmp`,
hashes it, and only then renames it onto `destPath` — atomic, and replacing
an existing file on both POSIX and Win32. An already-present destination is
re-hashed and re-fetched if it doesn't match, so a truncated cache heals
itself. With an empty `expectedSha256` the old behaviour is unchanged: an
existing non-empty file is left alone.

`downloadOcrModels` uses `defaultModelsBaseUrl()` when `baseUrl` is empty,
and looks up `knownSha256(name)` per file either way — a custom mirror
serving the stock file names still gets integrity checking.

SHA-256 is vendored in `model_downloader.cpp` (~70 lines, tested against the
NIST vectors including the 56-byte padding boundary). It is deliberately
**not** OpenSSL: no new dependency, `vcpkg.json` unchanged.

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
