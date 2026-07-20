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
    std::string modelType = "tiny";
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
