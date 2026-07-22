// Adapted from RapidAI/RapidOcrOnnx's AngleNet.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. Renamed to Classifier for arboOCR's public API.
#include "arboOCR/classifier.hpp"

#include <numeric>

#include <opencv2/imgproc.hpp>

#include "arboOCR/ocr_utils.hpp"
#include "ort_provider_utils.hpp"

namespace arbo::ocr {

namespace {

RawAngle scoreToAngle(const std::vector<float>& outputData) {
    int maxIndex = 0;
    float maxScore = 0.0f;
    for (size_t i = 0; i < outputData.size(); i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = static_cast<int>(i);
        }
    }
    return {maxIndex, maxScore};
}

} // namespace

Classifier::Classifier() = default;
Classifier::~Classifier() = default;

void Classifier::loadModel(const std::string& modelPath, bool useCuda,
                         bool useTensorrt, const std::string& trtCacheDir,
                         bool useFp16) {
    sessionOptions_.SetInterOpNumThreads(0);
    sessionOptions_.SetIntraOpNumThreads(0);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Input is always resized to kDstWidth x kDstHeight (192x48) before
    // inference, so min=opt=max — one shape, ever.
    detail::configureExecutionProviders(sessionOptions_, useCuda, useTensorrt, trtCacheDir,
        {"x:1x3x48x192", "x:1x3x48x192", "x:1x3x48x192"}, useFp16);

#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
#endif
    inputNamesPtr_ = detail::getInputNames(session_.get());
    outputNamesPtr_ = detail::getOutputNames(session_.get());
}

RawAngle Classifier::getAngle(cv::Mat& src) {
    if (!session_) {
        return {-1, 0.0f}; // guard: unloaded model
    }
    auto inputValues = substractMeanNormalize(src, meanValues_, normValues_);
    std::array<int64_t, 4> inputShape{1, src.channels(), src.rows, src.cols};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());

    std::vector<const char*> inputNames = {inputNamesPtr_.front().get()};
    std::vector<const char*> outputNames = {outputNamesPtr_.front().get()};
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outCount = std::accumulate(outShape.begin(), outShape.end(), int64_t{1}, std::multiplies<int64_t>());
    if (outCount <= 0) {
        return {-1, 0.0f}; // guard: unresolved/degenerate output shape
    }
    float* raw = outputTensors.front().GetTensorMutableData<float>();
    std::vector<float> outputData(raw, raw + outCount);
    return scoreToAngle(outputData);
}

std::vector<RawAngle> Classifier::getAngles(std::vector<cv::Mat>& partImages, bool doAngle, bool mostAngle) {
    size_t n = partImages.size();
    std::vector<RawAngle> angles(n);

    if (!doAngle) {
        for (size_t i = 0; i < n; i++) {
            angles[i] = {-1, 0.0f};
        }
        return angles;
    }

    for (size_t i = 0; i < n; i++) {
        if (partImages[i].empty()) {
            angles[i] = {-1, 0.0f}; // guard: e.g. a degenerate crop upstream (getRotateCropImage)
            continue;
        }
        cv::Mat resized;
        cv::resize(partImages[i], resized, cv::Size(kDstWidth, kDstHeight));
        angles[i] = getAngle(resized);
    }

    if (mostAngle) {
        double sum = 0.0;
        for (auto& a : angles) sum += a.index;
        int mostAngleIndex = (sum < static_cast<double>(angles.size()) / 2.0) ? 0 : 1;
        for (auto& a : angles) a.index = mostAngleIndex;
    }

    return angles;
}

} // namespace arbo::ocr
