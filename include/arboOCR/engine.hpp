#pragma once
// Orchestrates Detector + Classifier + Recognizer end-to-end
// (detect -> crop -> angle -> recognize). Adapted from RapidAI/RapidOcrOnnx's
// OcrLiteImpl::detect() pipeline — see THIRD_PARTY_NOTICES.md.

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "arboOCR/classifier.hpp"
#include "arboOCR/detector.hpp"
#include "arboOCR/preprocess.hpp"
#include "arboOCR/recognizer.hpp"
#include "arboOCR/types.hpp"

namespace arbo::ocr {

struct EngineConfig {
    std::string ocrVersion = "PP-OCRv6";
    // Selects the RECOGNIZER model only ("<ocrVersion>_rec_<modelType>.onnx").
    // Default "small": best CPU accuracy/latency tradeoff measured on SROIE
    // smoke compare (medium ~4× slower on CPU with no sim gain on that set).
    // Det is always "<ocrVersion>_det.onnx" unless detModelPath is set — modelType
    // does NOT select a det size variant.
    // Measured on PP-OCRv6 against the bundled sample receipt: "medium"
    // recognizer fixed real character errors "tiny" made (e.g. "Melavwai" ->
    // "Melawai", "Atasnama" -> "Atas nama") at ~5x the CPU latency (tiny
    // ~750ms -> medium ~3.9s on a Jetson Nano CPU fallback; TensorRT/CUDA
    // narrow this gap significantly). Prefer "medium" only with GPU headroom
    // after measuring on your data. See README's "Choosing a model size".
    std::string modelType = "small";
    float detBoxThresh = 0.5f;
    float detThresh = 0.3f;
    float detUnclipRatio = 1.6f;
    // Longest image side for det resize. 960 measured better than 1536 on the
// SROIE receipt smoke set (full-page sim ~87.7% vs ~85.9% with rec=small);
// larger limits can over-fragment or over-merge on dense receipts. Override
// for high-res pages if boxes look wrong.
int detLimitSideLen = 960;
    // Crops per recognition inference call (PaddleOCR/RapidOCR default: 6).
    // Raise on GPU/TensorRT when VRAM allows; lower on tight CPU budgets.
    // Values are clamped to >= 1. TensorRT engine profiles are built against
    // this value at Engine construction time.
    int recBatchNum = 6;
    bool useAngleCls = false;
    bool useCuda = false;
    bool useTensorrt = false;
    // TensorRT only: enable FP16 kernels (trt_fp16_enable). Default true —
    // this was previously always-on when TensorRT was selected. Set false
    // for FP32 engines when debugging accuracy. Changing this may require a
    // separate trtCacheDir or clearing the cache so engines are rebuilt.
    // INT8 is not supported (needs a calibration dataset).
    bool useFp16 = true;
    // ONNXRuntime thread-pool sizes, applied to all three sessions
    // (det/cls/rec). 0 means "let ORT decide", which in practice sizes the
    // pools for the whole machine. That is right for a process that owns the
    // box and wrong when it does not: N worker processes on one host each
    // spawn a machine-sized pool and thrash, and a container's CPU quota is
    // invisible to ORT. So this is a deployment knob (one worker per core,
    // cgroup limits), not a performance win — RapidOCR exposes the same pair
    // and states outright that bigger is not better, since the optimum is
    // workload-dependent. Measure before changing; ORT's default is usually
    // right for a process with the machine to itself. Negative values are
    // clamped to 0.
    int intraOpNumThreads = 0;  // 0 = ORT default
    int interOpNumThreads = 0;  // 0 = ORT default
    // Leave ORT's CPU memory arena enabled. false (default, current shipped
    // behavior): DisableCpuMemArena, bounded RSS (~135 MB on SROIE small) at
    // ~+17.6% engine latency (docs/improvement-roadmap.md item 1). true: ORT
    // default arena ON, faster, higher high-water RSS — matches oar-ocr.
    bool enableCpuMemArena = false;
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    // full image before detection. Off by default (matches
    // useAngleCls/useCuda/useTensorrt) — helps low-contrast documents
    // (faded receipts) recover text boxes DBNet would otherwise miss, but
    // adds per-image CPU cost and isn't universally beneficial (can
    // amplify noise on already-good scans). See preprocess.hpp::applyClahe.
    bool useClahe = false;
    // Split wide det boxes that look like two side-by-side fields (ink-gap
    // heuristic). Default off: on the SROIE smoke set, reading-order sort alone
    // closed most of the full-page gap (~87.7% → ~94.4%); aggressive split
    // over-fragmented and lost ~1 pt. Enable when det clearly fuses fields.
    bool splitOvermerged = false;
    // Drop det boxes whose area is at or below this, measured in DETECTOR
    // INPUT pixels — ppu-paddle-ocr's minimumAreaThreshold (default 20). A
    // box's source-image area is scaled by detInputArea/srcArea before the
    // comparison, because ppu tests the contour rect on its padded detector
    // tensor and this config keeps that meaning (so the default means the
    // same 20px^2 whatever detLimitSideLen does to an image). Boxes this
    // small hold no readable glyph, so dropping them saves a getRotateCropImage
    // plus a CRNN forward pass each, and keeps single-pixel det noise out of
    // the output entirely. 0 disables the filter (keeps every detected box).
    // Deliberately small: on the bundled SROIE receipt the smallest genuine
    // text line is orders of magnitude above it.
    float minDetBoxArea = 20.0f;
    // Drop lines whose recognition confidence is below this bar (Paddle
    // drop_score / ppu minimumConfidence). Symbol-only text uses bar+0.3.
    // 0 disables filtering (legacy RapidOcrOnnx keeps every box).
    float minimumConfidence = 0.5f;
    // Recover inter-word spaces the greedy CTC decode swallows: when the
    // space class (the dictionary's trailing " " key) is a strong runner-up
    // at an emitted character's timestep, emit the space too. Mirrors
    // ppu-paddle-ocr's spaceRecovery (default off there too). Off by default
    // because it can add spurious spaces in dense symbol runs — and because
    // this project's CTC output is already compared byte-for-byte against
    // fixtures, so it must stay opt-in. See Recognizer::setSpaceRecovery.
    bool spaceRecovery = false;
    // Populate LinePrediction::words with a polygon per word (per character for
    // CJK, which has no spaces to split on). Off by default: the spans it needs
    // are nearly free to compute, but carrying them for every line of every page
    // is not, and most callers only want line-level output.
    bool returnWordBoxes = false;
    std::string trtCacheDir = "models/trt_engines";
    std::string modelsDir = "models";
    // Fetch missing stock weights on Engine construction instead of throwing.
    // See ensureOcrModels() for exactly what this does and does not touch —
    // notably, a path you set explicitly below is never substituted. Set false
    // (or export ARBOOCR_OFFLINE=1) to forbid the process from touching the
    // network; a missing model is then the hard failure it used to be.
    bool autoDownload = true;
    // Directory URL to fetch from. Empty = defaultModelsBaseUrl(), which is
    // pinned to a release tag and checksum-verified. Point this at an internal
    // mirror or artifact store to keep the download inside your network;
    // stock file names are still verified against the built-in manifest.
    std::string modelsBaseUrl;
    // Optional absolute/relative paths. Empty = use modelsDir + default names
    // (see resolveModelPaths). Use these for custom/fine-tuned ONNX or dicts;
    // PP-OCRv6 default models are already multi-language (no language field).
    std::string detModelPath;
    std::string clsModelPath;
    std::string recModelPath;
    std::string dictPath;
};

struct ModelPaths {
    std::string det;
    std::string cls;
    std::string rec;
    std::string dict;
};

/// Resolve det/cls/rec/dict paths from config defaults and optional overrides.
/// Does not check that files exist. Pure / side-effect free.
ModelPaths resolveModelPaths(const EngineConfig& cfg);

/// Is this config allowed to touch the network for missing stock models?
/// True when `cfg.autoDownload` is set AND the process-wide `ARBOOCR_OFFLINE`
/// escape hatch is not engaged — engaged meaning the variable is set to a
/// non-empty value other than "0", so `ARBOOCR_OFFLINE=0` reads as an explicit
/// "stay online" rather than as offline.
///
/// The two inputs are answered here and only here. `ensureOcrModels()` gates
/// on this, and so must anything that *reports* on downloading (a CLI's
/// model-load diagnostic, say) — asking `cfg.autoDownload` directly gets the
/// flag but not the environment, and says the wrong thing to the user.
/// Reads the environment on every call rather than caching, so a process that
/// sets the variable late still sees it.
bool modelDownloadsAllowed(const EngineConfig& cfg);

/// resolveModelPaths(), then fetch whatever is missing — the network-touching
/// counterpart, called for you by the Engine constructor when
/// modelDownloadsAllowed() says so.
///
/// Per file, in order of precedence:
///   1. An explicitly set `cfg.*ModelPath` / `cfg.dictPath` is returned as-is,
///      always. A custom or fine-tuned model is never silently replaced by a
///      stock download just because the path is wrong.
///   2. An existing non-empty file under `cfg.modelsDir` wins — a populated
///      models directory means no network access at all.
///   3. Otherwise the stock file is downloaded from `cfg.modelsBaseUrl` (or
///      `defaultModelsBaseUrl()`) into `defaultModelsCacheDir()`, verified
///      against the built-in SHA-256 manifest, and that cache path is
///      returned.
///
/// `cls` is only considered when `cfg.useAngleCls` is set, and a dict failure
/// is non-fatal (the charset is often embedded in the rec ONNX). Never throws:
/// anything that cannot be fetched keeps its resolved-but-missing path, so the
/// caller gets the same model-load error it would have got anyway.
ModelPaths ensureOcrModels(const EngineConfig& cfg);

/// Auto-detect CUDA execution provider availability via ONNXRuntime.
bool detectCuda();

/// Auto-detect TensorRT execution provider availability via ONNXRuntime.
bool detectTensorrt();

/// Decode still-encoded image bytes (PNG/JPEG/... as they arrived over the
/// wire) into a BGR mat. Never throws: null data, zero size and undecodable
/// bytes all yield an empty Mat, mirroring how cv::imread reports an unreadable
/// path. cv::imdecode itself CV_Asserts on an empty buffer rather than
/// returning empty, so the guards below are load-bearing, not defensive noise.
/// Free function (not an Engine detail) so this contract stays unit-testable —
/// constructing an Engine needs real ONNX files on disk.
cv::Mat decodeImageBytes(const uint8_t* data, size_t size);

class Engine {
public:
    explicit Engine(const EngineConfig& config);

    /// "tensorrt", "cuda", or "cpu" — the backend actually selected after
    /// auto-detection.
    std::string backend() const { return backend_; }

    /// Run the full det -> crop -> angle -> recognize pipeline on one image
    /// loaded from disk. Never throws: image-read failures and inference
    /// exceptions both degrade to an empty-lines PagePrediction with
    /// elapsedMs set.
    PagePrediction recognize(const std::string& imagePath);

    /// Same pipeline as the path overload, but on an already-decoded BGR
    /// image in memory (e.g. a camera frame or API upload). Avoids a disk
    /// round-trip. Never throws; empty/invalid mats yield empty lines.
    /// `page.image` is left empty (no filename). The mat is not modified.
    PagePrediction recognize(const cv::Mat& image);

    /// Same pipeline again, but on *encoded* bytes (a PNG/JPEG upload, an FFI
    /// buffer, a socket read) — decoded via cv::imdecode instead of forcing
    /// callers to spill a temp file just to hand us bytes they already hold.
    /// Named recognizeEncoded rather than a recognize() overload because a
    /// bytes overload sitting next to recognize(const cv::Mat&) reads as
    /// "raw pixels" to every caller who skims it. Pointer + size is the
    /// primitive that binds to std::vector, std::string, std::span and foreign
    /// buffers with no copy at the boundary. Never throws: null data, zero size
    /// and garbage bytes all degrade to an empty-lines PagePrediction with
    /// elapsedMs set, exactly like an unreadable path.
    /// `page.image` is left empty — a byte buffer has no filename, and the Mat
    /// overload already emits `"image":""` for the same reason.
    PagePrediction recognizeEncoded(const uint8_t* data, size_t size);

    /// Non-blocking wrappers around recognize(). Each call launches work on
    /// a background thread and returns a future. Not safe to call
    /// concurrently on the same Engine instance (ONNXRuntime sessions are
    /// not shared-session concurrent) — use one outstanding async call at a
    /// time, or one Engine per worker. The Mat overload clones the image and
    /// the encoded overload copies the byte buffer, so the caller may
    /// free/reuse their buffer immediately.
    std::future<PagePrediction> recognizeAsync(const std::string& imagePath);
    std::future<PagePrediction> recognizeAsync(const cv::Mat& image);
    std::future<PagePrediction> recognizeEncodedAsync(const uint8_t* data, size_t size);

private:
    PagePrediction runPipeline(const cv::Mat& src, const std::string& imageName);

    Detector detector_;
    Classifier classifier_;
    Recognizer recognizer_;
    EngineConfig config_;
    std::string backend_ = "cpu";
};

} // namespace arbo::ocr
