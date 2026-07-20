#pragma once
// Adapted from RapidAI/RapidOcrOnnx's DbNet.h/.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. getTextBoxes/findRsBoxes are near-verbatim ports.
// Renamed to Detector for arboOCR's public API.

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "arboOCR/types.hpp"

namespace arbo::ocr {

class Detector {
public:
    Detector();
    ~Detector();

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    /// Load the DBNet ONNX model. `useCuda` appends CUDA EP,
    /// `useTensorrt` appends TensorRT EP with FP16 + profile shapes.
    void loadModel(const std::string& modelPath, bool useCuda = false,
                   bool useTensorrt = false, const std::string& trtCacheDir = "");

    /// Run detection on `src` (already resized to `scale.dstWidth x
    /// scale.dstHeight` internally). Returns text boxes in `src`'s original
    /// (un-scaled) coordinate space.
    std::vector<RawTextBox> getTextBoxes(
        const cv::Mat& src,
        const ScaleParam& scale,
        float boxScoreThresh,
        float boxThresh,
        float unClipRatio
    );

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "Detector"};
    Ort::SessionOptions sessionOptions_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;

    // ImageNet-style normalization (matches RapidOcrOnnx's DbNet.h constants,
    // which match PaddleOCR's det model preprocessing).
    const float meanValues_[3] = {0.485f * 255, 0.456f * 255, 0.406f * 255};
    const float normValues_[3] = {1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f};
};

} // namespace arbo::ocr
