#pragma once
// Orchestrates Detector + Classifier + Recognizer end-to-end
// (detect -> crop -> angle -> recognize). Adapted from RapidAI/RapidOcrOnnx's
// OcrLiteImpl::detect() pipeline — see THIRD_PARTY_NOTICES.md.

#include <string>
#include <vector>

#include "arboOCR/classifier.hpp"
#include "arboOCR/detector.hpp"
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
    // slow for your latency budget. See README's "Model size trade-off"
    // section for the full comparison.
    std::string modelType = "medium";
    float detBoxThresh = 0.5f;
    float detThresh = 0.3f;
    float detUnclipRatio = 1.6f;
    int detLimitSideLen = 1536;
    bool useAngleCls = false;
    bool useCuda = false;
    bool useTensorrt = false;
    std::string trtCacheDir = "models/trt_engines";
    std::string modelsDir = "models";
};

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

    /// Run the full det -> crop -> angle -> recognize pipeline on one image.
    /// Never throws: image-read failures and inference exceptions both
    /// degrade to an empty-lines PagePrediction with elapsedMs set.
    PagePrediction recognize(const std::string& imagePath);

private:
    Detector detector_;
    Classifier classifier_;
    Recognizer recognizer_;
    EngineConfig config_;
    std::string backend_ = "cpu";
};

} // namespace arbo::ocr
