#pragma once
// Orchestrates Detector + Classifier + Recognizer end-to-end
// (detect -> crop -> angle -> recognize). Adapted from RapidAI/RapidOcrOnnx's
// OcrLiteImpl::detect() pipeline — see THIRD_PARTY_NOTICES.md.

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
    // Selects the RECOGNIZER model only ("<ocrVersion>_rec_<modelType>.onnx")
    // — the detector is always loaded from "<ocrVersion>_det.onnx" (no size
    // suffix), so this does not affect detection. Measured on PP-OCRv6
    // against the bundled sample receipt: "medium" recognizer fixed real
    // character errors "tiny" made (e.g. "Melavwai" -> "Melawai",
    // "Atasnama" -> "Atas nama") at ~5x the CPU latency (tiny ~750ms ->
    // medium ~3.9s on a Jetson Nano CPU fallback; TensorRT/CUDA narrow this
    // gap significantly). "small" is a middle ground if "medium" is too
    // slow for your latency budget. See README's "Choosing a model size"
    // section for the full comparison.
    std::string modelType = "medium";
    float detBoxThresh = 0.5f;
    float detThresh = 0.3f;
    float detUnclipRatio = 1.6f;
    int detLimitSideLen = 1536;
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
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    // full image before detection. Off by default (matches
    // useAngleCls/useCuda/useTensorrt) — helps low-contrast documents
    // (faded receipts) recover text boxes DBNet would otherwise miss, but
    // adds per-image CPU cost and isn't universally beneficial (can
    // amplify noise on already-good scans). See preprocess.hpp::applyClahe.
    bool useClahe = false;
    std::string trtCacheDir = "models/trt_engines";
    std::string modelsDir = "models";
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

/// Auto-detect CUDA execution provider availability via ONNXRuntime.
bool detectCuda();

/// Auto-detect TensorRT execution provider availability via ONNXRuntime.
bool detectTensorrt();

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

    /// Non-blocking wrappers around recognize(). Each call launches work on
    /// a background thread and returns a future. Not safe to call
    /// concurrently on the same Engine instance (ONNXRuntime sessions are
    /// not shared-session concurrent) — use one outstanding async call at a
    /// time, or one Engine per worker. The Mat overload clones the image so
    /// the caller may free/reuse their buffer immediately.
    std::future<PagePrediction> recognizeAsync(const std::string& imagePath);
    std::future<PagePrediction> recognizeAsync(const cv::Mat& image);

private:
    PagePrediction runPipeline(const cv::Mat& src, const std::string& imageName);

    Detector detector_;
    Classifier classifier_;
    Recognizer recognizer_;
    EngineConfig config_;
    std::string backend_ = "cpu";
};

} // namespace arbo::ocr
