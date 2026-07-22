#pragma once
// Adapted from RapidAI/RapidOcrOnnx's AngleNet.h/.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. Renamed to Classifier for arboOCR's public API.

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "arboOCR/types.hpp"

namespace arbo::ocr {

class Classifier {
public:
    Classifier();
    ~Classifier();

    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;

    void loadModel(const std::string& modelPath, bool useCuda = false,
                   bool useTensorrt = false, const std::string& trtCacheDir = "",
                   bool useFp16 = true);

    /// Classify each cropped text-line image's orientation (0 = upright,
    /// 1 = 180-degree rotated). If `doAngle` is false, returns index=-1 for
    /// every image without touching the model — a genuine no-op skip, not
    /// just a disabled post-filter.
    /// If `mostAngle` is true, all images are forced to the majority-vote
    /// angle across the batch (reduces flicker on noisy per-line angle
    /// classification for a single scanned page).
    std::vector<RawAngle> getAngles(std::vector<cv::Mat>& partImages, bool doAngle, bool mostAngle);

private:
    RawAngle getAngle(cv::Mat& src);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "Classifier"};
    Ort::SessionOptions sessionOptions_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;

    static constexpr int kDstWidth = 192;
    static constexpr int kDstHeight = 48;
    const float meanValues_[3] = {127.5f, 127.5f, 127.5f};
    const float normValues_[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
};

} // namespace arbo::ocr
