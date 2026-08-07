# arboOCR Improvement Roadmap (2026-08-07)

Gap analysis of arboOCR against **RapidOCR v3.9.x** as documented at
<https://rapidai.github.io/RapidOCRDocs/> (local mirror: `../RapidOCRDocs/docs/`).
RapidOCR is the mature Python OCR library wrapping the same PP-OCR model family
arboOCR uses, and — unlike most OCR projects — it publishes *measured* numbers for
its engine and parameter choices. That makes it a useful yardstick: where RapidOCR
measured something and arboOCR guessed, the measurement wins. Every item below
carries the arboOCR `file:line` it applies to, verified against the tree at the
date above. This document is a checklist, not a promise of ordering.

## Summary

| # | Item | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | Disable ORT CPU memory arena | High — 5.6 GB RSS on Jetson | S | **done** (`8c01bcb`) |
| 2 | Stop upscaling small images in `getScaleParam` | High — wasted detector cost | S | **done** (`8c01bcb`) |
| 3 | `downloadOcrModels` never fetches `_dict.txt` | High — documented path is broken | S | **done** (`8c01bcb`) |
| 4 | `Engine(cfg)` crashes the CLI on a bad model | High — first call every user makes | S | **done** (`8c01bcb`) |
| 5 | No byte-buffer (encoded-bytes) input | Medium — temp file per web request | S | **done** (`37107b3`) |
| 6 | One image per process spawn | Medium — 200 model loads for 200 pages | M | open |
| 7 | Expose accuracy-tuning flags on the CLI | High — Go/Rust/PHP are locked out | S | **done** (`8c01bcb`) |
| 8 | Word / char-level boxes | Medium — data already computed, then dropped | S | **done** (`HEAD`) |
| 9 | Visualization helper | Low — README currently hand-waves it | S | **done** (`37107b3`) |
| 10 | Markdown / text-layout export | Low — #8 done, now unblocked | M | **done** (`HEAD`) |
| 11 | Tunable intra/inter-op thread counts | Low — workload-dependent | S | open |
| 12 | CMake `install()` / package export | Medium — contradicts the README pitch | M | **done** (`37107b3`) |
| 13 | `sortLinesReadingOrder` hard-codes a 12px y-tolerance | Medium — breaks on high-DPI scans | S | **done** (`8c01bcb`) |
| 14 | No test covers the TensorRT path or the FP16 flag | Medium — unvalidated default | M | open — harness ready (`scripts/fp16_ab.py`), Jetson run 2026-08-10 |

Effort: **S** = under a day, **M** = a few days, **L** = a week or more.

Eleven of fourteen are done. Remaining: **#6** batch input, **#11** thread controls, and **#14** the
TensorRT/FP16 validation — a risk rather than a gap, since `useFp16 = true`
ships as the default and nothing exercises it. The dev machine has no CUDA
(AMD RX 6600), so #14 runs on the Jetson on **2026-08-10**.

---

## Tier 1 — small diffs, measured payoff

### 1. ORT CPU memory arena is left at ORT's default (ON)

**What's wrong.** All three sessions configure only thread counts and the
optimization level; nothing touches the CPU memory arena, so ORT's default
(`enable_cpu_mem_arena = True`) applies.

- `src/arboOCR/detector.cpp:69-71`
- `src/arboOCR/recognizer.cpp:39-41`
- `src/arboOCR/classifier.cpp:36-38`

**Evidence.** RapidOCR profiled a single inference both ways:

| `enable_cpu_mem_arena` | Peak RSS | Delta on the inference line |
|---|---|---|
| `True` | 5695.5 MiB | +5618.3 MiB |
| `False` | 82.1 MiB | +5.3 MiB |

A 5618 MB difference on **one** inference, and the arena never releases it. The
purchase price of that memory is roughly 13% inference latency. RapidOCR ships
`enable_cpu_mem_arena: false` as its shipped default (`parameters.md:113`) —
they made this call already.

Jetson Nano is arboOCR's stated target and its benchmark machine
(`docs/jetson-verification.md`). On a 4 GB Nano this is not a tuning knob; it is
the difference between running and being OOM-killed.

**Fix.** In each of the three `loadModel()` bodies:

```cpp
opts.DisableCpuMemArena();
opts.AddConfigEntry("session.arena_extend_strategy", "kSameAsRequested");
```

Optionally gate behind an `EngineConfig` bool defaulting to *disabled* — matching
RapidOCR — rather than defaulting to ORT's arena-on behaviour.

### 2. `getScaleParam` upscales small images

**What's wrong.** `getScaleParam` (`src/arboOCR/ocr_utils.cpp:19-40`) computes
`ratio = targetSize / longSide` unconditionally and then floors both dimensions
to a multiple of 32. There is no `ratio > 1` guard, so a small image is
**upscaled** to `detLimitSideLen`. With the default `detLimitSideLen = 960`
(`include/arboOCR/engine.hpp:41`), a 200px thumbnail becomes a 960px detector
input and pays full detection cost for invented pixels.

**Evidence.** RapidOCR treats this as a first-class parameter rather than a
hardcoded policy: `Det.limit_type` takes `min | max` and `Det.limit_side_len`
defaults to 736 (`parameters.md:248-249, 277-279`). Their default `min` also
scales small images up — the point is that `max` exists and is one config field
away. arboOCR has no such switch at all.

**Fix.**
- Minimal: clamp `ratio` to `<= 1.0f`. One line, no API change, no config field.
- Proper: add `EngineConfig::detLimitType` (`min`/`max`) mirroring RapidOCR, and
  thread it into `getScaleParam`.

### 3. `downloadOcrModels` never fetches the dictionary

**What's wrong.** `downloadOcrModels` (`src/arboOCR/model_downloader.cpp:65`)
builds a three-entry file list at `:73-77` — `_det.onnx`, `_cls.onnx`,
`_rec_<modelType>.onnx`. `resolveModelPaths` (`src/arboOCR/engine.cpp:52-68`)
resolves a fourth path, `<ocrVersion>_rec_<modelType>_dict.txt` (`:64-66`), and
`Engine`'s constructor falls back to it whenever the ONNX carries no `character`
metadata (`src/arboOCR/engine.cpp:91-99`).

So README "Option B — download programmatically" (`README.md:231-243`) produces a
models directory that is unusable unless the ONNX happens to embed its character
list. When it doesn't, the engine logs `Recognizer has no character dictionary
loaded` and every recognition comes back empty.

There is also no checksum verification and no progress callback anywhere in
`model_downloader.cpp`. RapidOCR pins a SHA256 per model in its
`default_models.yaml` for exactly this reason.

**Fix.** Add the `_dict.txt` entry to the file list; treat its 404 as non-fatal
(some ONNX genuinely carry metadata) but surface it in the `DownloadResult`. Add
an optional expected-SHA256 argument and a progress callback while the signature
is being touched anyway.

### 4. `Engine(cfg)` throws uncaught and takes the CLI down with it

**What's wrong.** A missing or corrupt model file makes
`detector_.loadModel(...)` at `src/arboOCR/engine.cpp:85` throw `Ort::Exception`
out of the `Engine` constructor. `cli/arboocr_demo.cpp:68` constructs the engine
with no guard:

```cpp
arbo::ocr::Engine engine(cfg);
```

The exception unwinds past `main()` into `std::terminate()`. The CLI already
knows this failure mode is ugly — it catches cxxopts parse errors twenty lines
earlier for precisely this reason (`cli/arboocr_demo.cpp:39-46`, noting that the
hardened runtime turns it into a bogus `STATUS_STACK_BUFFER_OVERRUN`).

This also quietly undoes the guarantee the codebase maintains everywhere else.
`recognize()` never throws — it is documented at `include/arboOCR/engine.hpp:109-111`,
implemented with a catch-all at `src/arboOCR/engine.cpp:156-159`, and the CLI
comments on it at `:83-84`. Construction is the one call every user makes first,
and it is the one that crashes.

The native Python bindings are better off by accident: pybind11 translates
`std::exception` into a Python `RuntimeError`, so `python/bindings/module.cpp:114`
degrades to a catchable exception with an ORT-flavoured message rather than a
crash. The subprocess wrappers see only a non-zero exit and stderr noise.

**Fix.** Wrap the `Engine` construction in `cli/arboocr_demo.cpp` in a
`try/catch (const std::exception&)` that prints the model paths from
`resolveModelPaths(cfg)` plus `e.what()`, and returns 1. Consider a
`Engine::tryCreate(cfg) -> std::optional<Engine>` (or an `std::expected`-shaped
result) so C++ consumers get a non-throwing path too.

---

## Tier 2 — the wrapper ecosystem is locked out of its own library

Three of the four out-of-tree wrappers — `arbo-ocr-go`, `arbo-ocr-rust`,
`arbo-ocr-php` — drive `arboocr_demo` as a subprocess and parse its `--json`
stdout. So does the `arbo-ocr-python` PyPI package. For all of them, **the CLI is
the real public API**, and the CLI exposes a strict subset of `EngineConfig`.

`cli/arboocr_demo.cpp:14-34` exposes: `--image --models-dir --ocr-version
--model-type --angle --cuda --tensorrt --fp16 --clahe --json --det-model
--cls-model --rec-model --dict`.

It does **not** expose: `detBoxThresh`, `detThresh`, `detUnclipRatio`,
`detLimitSideLen`, `recBatchNum`, `splitOvermerged`, `minimumConfidence`,
`trtCacheDir`, or log level — every one of which is a real field on
`EngineConfig` (`include/arboOCR/engine.hpp:34-72`).

The exception is native Python: arboOCR's in-tree pybind11 module already binds
the full config surface (`python/bindings/module.cpp:90-111`), including
`det_box_thresh`, `det_limit_side_len`, `rec_batch_num`, `split_overmerged`,
`minimum_confidence` and `trt_cache_dir`. Go, Rust and PHP have no such path.
The entire accuracy-tuning story from the recent accuracy cycle is unreachable
from three of the five language targets.

### 5. No byte-buffer input

`cv::imdecode` appears nowhere in the tree. `Engine::recognize(const std::string&)`
(`src/arboOCR/engine.cpp:166-173`) only calls `cv::imread` on a path. There *is*
an in-memory overload, `Engine::recognize(const cv::Mat&)` at `:175-177`, but it
takes an already-decoded BGR mat — which is useful from C++ and from the Python
bindings (numpy path, `python/bindings/module.cpp:46-52`) and useless from a
subprocess wrapper holding a PNG upload.

Every web/API integration therefore writes a temp file per request.

**Fix.** Add `PagePrediction recognize(const uint8_t* data, size_t len)` (or a
`std::span<const std::byte>` overload) that runs `cv::imdecode` and forwards to
`runPipeline`. Expose it from the bindings as a `bytes` overload. For the CLI, a
`--stdin-image` mode reading encoded bytes from stdin covers the wrappers.

### 6. One image per process spawn

The CLI takes exactly one `--image` and exits. A 200-page job therefore pays 200
process spawns **and** 200 full model loads — and on the TensorRT path, 200
engine-cache lookups. The models are the expensive part; the spawn is merely
insulting.

**Fix.** Either `--images-dir <dir>` or (better, composes with shell pipelines)
reading newline-separated paths from stdin and emitting one JSON object per line.
Both amortize the model load across the batch. The wrappers can then keep one
long-lived process instead of one per page.

---

## Tier 3 — genuine feature gaps vs RapidOCR

### 7. Word / character-level boxes

RapidOCR exposes `return_word_box` and `return_single_char_box`
(`parameters.md:35-36, 72-87`).

arboOCR **already computes** the underlying data. The recognizer decodes
per-token with CTC timestep positions and per-character scores
(`src/arboOCR/recognizer.cpp:131-156`) — that is exactly how `injectGapSpaces`
knows where to insert spaces (`src/arboOCR/ocr_utils.cpp:473`,
`include/arboOCR/ocr_utils.hpp:74-76`). Then `src/arboOCR/engine.cpp:141` averages
the whole vector into one float via `meanRecScore` and the positions are
discarded.

**Fix.** Keep `TextLine::charScores` (`include/arboOCR/types.hpp:36`) and the
normalized timestep positions on `LinePrediction` behind a
`EngineConfig::returnWordBox` flag; map normalized positions back to crop-space,
then to page-space through the existing box geometry. Low effort for the value,
and it is the hard prerequisite for item 9.

### 8. No visualization helper

RapidOCR ships `.vis("out.jpg")` on every result object — it appears in roughly
every usage example in their docs. arboOCR's README tells the user to draw the
boxes themselves (`README.md:74`, `examples/basic_recognize.cpp:58`: "draw boxes
with e.g. `cv::polylines`"). That is a suggestion, not an API.

**Fix.** `cv::Mat visualize(const cv::Mat& src, const PagePrediction&)` in a new
`arboOCR/visualize.hpp`, plus `--vis <path>` on the CLI. OpenCV is already a hard
dependency, so this adds nothing to the dependency surface.

### 9. No markdown / text-layout export

RapidOCR has `result.to_markdown()`, added in `rapidocr>=3.2.0` and described in
their own docs as *rough* (`how_to_convert_to_markdown.md:7`); the rules were
last revised in RapidOCR PR #672 (`changelog/v3.8.2.md:11`).

Needs item 7 first — you cannot infer column structure from line-level boxes
alone. Worth doing only after word boxes land, and worth copying their honesty
about it being approximate.

### 10. Thread counts are hard-coded

All three sessions set `SetInterOpNumThreads(0)` and `SetIntraOpNumThreads(0)`
(`detector.cpp:69-70`, `recognizer.cpp:39-40`, `classifier.cpp:36-37`). Zero means
"ORT decides", which is a reasonable default and a bad *only* option.

RapidOCR exposes both (`parameters.md:111-112`, defaults `-1`) and states
explicitly that bigger is not better — the optimum is workload-dependent, and on
a shared or containerized host the right answer is often 1 or 2.

**Fix.** `EngineConfig::intraOpNumThreads` / `interOpNumThreads`, defaulting to 0
(current behaviour), threaded through `loadModel` and exposed on the CLI.

### 11. No CMake install / export

`arboOCR` is declared `STATIC` at `CMakeLists.txt:41` and that is the end of it —
the tree contains no `install()`, no `export()`, no
`configure_package_config_file`, and no `.pc` file. There is no way to
`find_package(arboOCR)`; consumers must vendor the entire source tree, which
directly contradicts the README's "drop it into your own project" framing.

**Fix.** `install(TARGETS arboOCR arboocr_clipper EXPORT arboOCRTargets ...)`,
`install(DIRECTORY include/ ...)`, a generated `arboOCRConfig.cmake` with the
OpenCV/CURL/onnxruntime `find_dependency` calls, and an `arboOCRConfigVersion.cmake`.
Note the ORT link path is conditional on `ARBOOCR_USE_SYSTEM_DEPS`
(`CMakeLists.txt:56-64`), so the exported config has to handle both branches —
that is the part that makes this M rather than S.

### 12. `sortLinesReadingOrder` hard-codes a 12px y-tolerance

`sortLinesReadingOrder` (`src/arboOCR/ocr_utils.cpp:369-390`) groups lines into
visual rows with a fixed tolerance:

```cpp
// Same visual row if y within ~half a typical line — use 12px fallback.
const float yTol = 12.f;
```

The comment (`:383`) already concedes it is a fallback for "half a typical line".
12px is half a typical line at roughly 100-150 DPI. At 600 DPI it is a quarter of
a character, and every row on the page fragments into one-cell rows in arbitrary
x-order. This matters: the accuracy cycle measured reading-order sort alone as
worth ~6.7 points of full-page similarity on the SROIE set
(`include/arboOCR/engine.hpp:64-66`), and that entire gain is resolution-dependent
in its current form.

**Fix.** Compute the median polygon height across `lines` and use
`yTol = 0.5f * medianHeight`, clamping to a small floor for degenerate inputs.
Keep 12px only when `lines` is too small to estimate from.

---

## Risk — validate, do not assume

### 13. `useFp16 = true` is the TensorRT default and is untested

arboOCR defaults `EngineConfig::useFp16 = true` (`include/arboOCR/engine.hpp:55`)
and `--fp16` defaults to `true` on the CLI (`cli/arboocr_demo.cpp:23-24`).

**Evidence that this is worth checking.** RapidOCR measured TensorRT FP16
catastrophically breaking the PP-OCR **server** detection models:

| Model | Baseline | Baseline H-mean | TensorRT FP16 H-mean |
|---|---|---|---|
| `ch_PP-OCRv4_det_server` | TensorRT FP32 | 0.8161 | **0.0905** |
| `ch_PP-OCRv5_det_server` | ONNX Runtime | 0.7883 | **0.2751** |

(`support_tensorrt.md:283-284, 306-312`.) Both reproduce on an A800 and an RTX
3060, so it is not a single-card artifact. RapidOCR's own note at `:314` says the
cause is undiagnosed. Mobile detection models and **all** recognition models were
unaffected — v5 rec server moved 0.8161 → 0.8129, i.e. noise.

arboOCR ships a single non-server detector (`PP-OCRv6_det.onnx`, no size
variants — `README.md:247-249`), so it is probably in the safe class. "Probably"
is doing real work in that sentence, and there is currently **no test covering the
TensorRT path or the FP16 flag at all**: `tests/test_engine.cpp:22` asserts the
default is `true` and `:165-166` checks that the string `"tensorrt"` survives JSON
serialization. Neither runs an engine.

Also worth knowing before recommending FP16: TensorRT FP16 engine **builds** are
substantially slower than FP32 on the same model (A800, `support_tensorrt.md:64-101`):

| Model | FP32 build | FP16 build | Ratio |
|---|---|---|---|
| `ch_PP-OCRv5_det_mobile` | 407.53 s | 1438.65 s | 3.53x |
| `ch_PP-OCRv4_det_server` | 109.56 s | 299.10 s | 2.73x |
| `ch_PP-OCRv5_rec_mobile` | 543.68 s | 1437.16 s | 2.64x |
| `ch_PP-OCRv4_det_mobile` | 315.53 s | 521.02 s | 1.65x |
| `ch_PP-OCRv4_rec_mobile` | 480.52 s | 603.08 s | 1.26x |

On a Jetson Nano — where the existing verification run already describes the
first TensorRT run as "slow, minutes" (`docs/jetson-verification.md:36-37`) — a
2-3x multiplier on cold-start engine build is a real cost that FP16 must earn
back.

### Harness: `scripts/fp16_ab.py`

The A/B is written and self-tested, but **has not been run against real
TensorRT** — no NVIDIA GPU is available on the development machine (the vcpkg
onnxruntime build ships CPU only: no `onnxruntime_providers_tensorrt.dll`), and
the Jetson is currently unreachable through its cloudflared tunnel
(`websocket: bad handshake`). Item #14 stays **open** until someone runs this on
the Nano.

It needs no ground-truth labels. CPU is the well-tested path, so it asks the
question the default actually rides on: *does TensorRT agree with CPU, and does
FP16 agree less than FP32?* Each arm gets its own `--trt-cache-dir`, because
mixing them loads an engine built for the other precision and makes the whole
comparison meaningless.

```bash
python scripts/fp16_ab.py --images path/to/receipts --models-dir models \
    --bin build/jetson/arboocr_demo --model-type small
```

Exit codes: `0` pass, `1` FP16 diverged past `--fail-under` (default 0.98),
`2` inconclusive.

Note what `2` is for. `Engine` auto-falls back TensorRT → CUDA → CPU, so on a
machine without TensorRT every arm runs identical CPU code and agrees with
itself 100% — a confident-looking PASS that proves nothing. The script reads
`backend` out of the demo's JSON and refuses to grade unless the `trt-*` arms
actually got `tensorrt`. That guard is the only reason its output can be
trusted; the first version of this script did not have it and printed exactly
that false PASS.

The existing evidence covers less than it appears to:
`docs/jetson-verification.md:31-36` reports CPU and TensorRT producing identical
text, but that was **tiny** models on **one** receipt, and the library default is
`small`. Run the harness at `--model-type small` and `medium` before treating
`useFp16 = true` as validated.

**Fix.** One A/B on the SROIE smoke set: `--tensorrt --fp16` vs
`--tensorrt --fp16=false`, comparing per-line text and full-page similarity. If
they match, document the result and keep the default. If they diverge, flip the
default to FP32 and make FP16 opt-in. Either way, add a test that at minimum
constructs an engine on both paths when TensorRT is available and skips
otherwise — mirroring the existing optional-model-inference skip.

---

## Deliberately not doing

RapidOCR's engine matrix looks impressive. By their own measurements, most of it
is not worth having — and the parts that are come at the cost of arboOCR's
single-runtime design.

| Not doing | Why, per RapidOCR's own numbers |
|---|---|
| **CoreML EP** | 3.16x–14.04x *slower* than the CPU provider on a MacBook Pro M2. PP-OCRv5 rec mobile: 18.50 ms CPU vs 259.70 ms CoreML. Accuracy bit-identical, so there is nothing to trade for. RapidOCR keeps the provider on the bet that Apple improves it later, not on measured merit. |
| **OpenVINO** | Genuinely the fastest CPU engine they measured — PP-OCRv6 medium det 0.4476 s vs ONNX Runtime 0.9491 s, same H-mean. But it does not release memory after large images ([openvino#11939](https://github.com/openvinotoolkit/openvino/issues/11939)), which is the exact failure mode item 1 exists to avoid, and adopting it means abandoning the single-runtime design. |
| **MNN** | Faster on detection (v4 det mobile 0.182 → 0.159 s), *slower* on several recognition models (v4 rec mobile 0.0176 → 0.0213 s; v5 rec mobile 0.0196 → 0.0373 s; v5 rec server 0.0582 → 0.0724 s). Identical accuracy throughout. A wash, in exchange for a second runtime dependency. |
| **ONNXRuntime CUDA EP** | Measured *slower* than CPU on both a GTX 1660 Super (2.574 vs 1.183 s/img) and an RTX 3090 (0.999 vs 0.505 s/img), because OCR detection is inherently dynamic-shaped. RapidOCR's conclusion is a flat "not recommended". arboOCR's `useCuda = false` default (`include/arboOCR/engine.hpp:48`) is already correct — the work here is documenting *why*, not changing it. |
| **PyTorch / PaddlePaddle backends** | Python-ecosystem-bound. Irrelevant to a C++ library. |
| **DirectML** | RapidOCR has a how-to but zero benchmark data. Nothing to evaluate against. |
| **INT8 quantization** | Needs a representative calibration set and is easy to get badly wrong for multilingual text. Already correctly deferred — `include/arboOCR/engine.hpp:54` says so explicitly. |
| **HTTP server, web UI, layout analysis, table recognition** | Separate repos even inside RapidAI's own ecosystem (`rapidocr_api`, `rapidocr_web`). arboOCR's pitch is a single-purpose inference core. Keep it that way. |

---

## Evidence sources

All RapidOCR measurements above come from these pages (paths relative to
`https://rapidai.github.io/RapidOCRDocs/`, mirrored locally under
`../RapidOCRDocs/docs/`):

| Claim | Page |
|---|---|
| CPU memory arena: 5695.5 vs 82.1 MiB, ~13% latency | `blog/posts/inference_engine/onnxruntime/infer_optim.md` |
| `limit_type` / `limit_side_len`, `intra_op_num_threads`, `enable_cpu_mem_arena` default, `return_word_box`, `return_single_char_box` | `install_usage/rapidocr/parameters.md` |
| TensorRT FP16 H-mean collapse on server det; FP16 vs FP32 engine build times | `blog/posts/inference_engine/support_tensorrt.md` |
| CoreML vs CPU provider latency and accuracy on M2 | `blog/posts/inference_engine/compare_coreml_cpu_provider_perf.md` |
| ONNXRuntime CUDA EP slower than CPU (GTX 1660S, RTX 3090) | `blog/posts/inference_engine/onnxruntime/onnxruntime-gpu.md` |
| OpenVINO memory-not-released (issue #11939) | `blog/posts/inference_engine/openvino/infer.md` |
| OpenVINO vs ONNX Runtime on PP-OCRv6 det (0.4476 vs 0.9491 s) | `blog/posts/inference_engine/support_PP-OCRv6_det_engines.md` |
| MNN det/rec latency comparison | `blog/posts/inference_engine/support_mnn_engine.md` |
| `to_markdown()` exists and is "rough" | `install_usage/rapidocr/how_to_convert_to_markdown.md` |
| `to_markdown` rules revised in PR #672 | `changelog/v3.8.2.md` |
| `.vis()` on result objects | `install_usage/rapidocr/how_to_use_infer_engine.md` (and most usage examples) |
| Per-model SHA256 pinning for downloads | `blog/posts/inference_engine/support_PP-OCRv6_det_engines.md` |
