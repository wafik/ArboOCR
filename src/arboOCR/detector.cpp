// Adapted from RapidAI/RapidOcrOnnx's DbNet.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. Renamed to Detector for arboOCR's public API.
#include "arboOCR/detector.hpp"

#include <numeric>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

#include "arboOCR/ocr_utils.hpp"

namespace arbo::ocr {

namespace {

std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session* session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> names;
    size_t n = session->GetInputCount();
    names.reserve(n);
    for (size_t i = 0; i < n; i++) {
        names.push_back(session->GetInputNameAllocated(i, allocator));
    }
    return names;
}

std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session* session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> names;
    size_t n = session->GetOutputCount();
    names.reserve(n);
    for (size_t i = 0; i < n; i++) {
        names.push_back(session->GetOutputNameAllocated(i, allocator));
    }
    return names;
}

std::vector<RawTextBox> findRsBoxes(
    const cv::Mat& predMat, const cv::Mat& dilateMat, const ScaleParam& s,
    float boxScoreThresh, float unClipRatio
) {
    const int longSideThresh = 3;
    const size_t maxCandidates = 1000;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dilateMat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    size_t numContours = std::min(contours.size(), maxCandidates);
    std::vector<RawTextBox> rsBoxes;

    for (size_t i = 0; i < numContours; i++) {
        if (contours[i].size() <= 2) continue;
        cv::RotatedRect minAreaRect = cv::minAreaRect(contours[i]);

        float longSide = 0.0f;
        auto minBoxes = getMinBoxes(minAreaRect, longSide);
        if (longSide < longSideThresh) continue;

        float boxScore = boxScoreFast(minBoxes, predMat);
        if (boxScore < boxScoreThresh) continue;

        cv::RotatedRect clipRect = unClipBox(minBoxes, unClipRatio);
        if (clipRect.size.height < 1.001f && clipRect.size.width < 1.001f) continue;

        auto clipMinBoxes = getMinBoxes(clipRect, longSide);
        if (longSide < longSideThresh + 2) continue;

        std::vector<cv::Point> intBox;
        for (auto& pt : clipMinBoxes) {
            float x = pt.x / s.ratioWidth;
            float y = pt.y / s.ratioHeight;
            int ptX = std::min(std::max(static_cast<int>(x), 0), s.srcWidth - 1);
            int ptY = std::min(std::max(static_cast<int>(y), 0), s.srcHeight - 1);
            intBox.emplace_back(ptX, ptY);
        }
        rsBoxes.push_back(RawTextBox{intBox, boxScore});
    }
    std::reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

} // namespace

Detector::Detector() = default;
Detector::~Detector() = default;

void Detector::loadModel(const std::string& modelPath, bool useCuda,
                       bool useTensorrt, const std::string& trtCacheDir) {
    sessionOptions_.SetInterOpNumThreads(0);
    sessionOptions_.SetIntraOpNumThreads(0);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // TensorRT must be appended BEFORE CUDA (ORT tries providers in order)
    if (useTensorrt) {
        Ort::TensorRTProviderOptions trtOpts;
        std::unordered_map<std::string, std::string> opts = {
            {"device_id", "0"},
            {"trt_max_workspace_size", "1073741824"}, // 1GB
            {"trt_fp16_enable", "1"},
            {"trt_engine_cache_enable", "1"},
            // Fixed profile range covers detLimitSideLen (default 1536) and
            // larger inputs without triggering an ORT TRT rebuild per image
            // shape (see TENSORRT_ENGINE_PORT_PLAN.md Opsi 1).
            {"trt_profile_min_shapes", "x:1x3x32x32"},
            {"trt_profile_opt_shapes", "x:1x3x1536x1536"},
            {"trt_profile_max_shapes", "x:1x3x2048x2048"},
        };
        if (!trtCacheDir.empty()) {
            opts["trt_engine_cache_path"] = trtCacheDir;
        }
        trtOpts.Update(opts);
        sessionOptions_.AppendExecutionProvider_TensorRT_V2(*trtOpts);
    }

    if (useCuda) {
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOptions);
    }

#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
#endif
    inputNamesPtr_ = getInputNames(session_.get());
    outputNamesPtr_ = getOutputNames(session_.get());
}

std::vector<RawTextBox> Detector::getTextBoxes(
    const cv::Mat& src, const ScaleParam& scale,
    float boxScoreThresh, float boxThresh, float unClipRatio
) {
    if (!session_) {
        return {}; // guard: unloaded model returns empty, never crashes callers
    }

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(scale.dstWidth, scale.dstHeight));
    auto inputValues = substractMeanNormalize(resized, meanValues_, normValues_);
    std::array<int64_t, 4> inputShape{1, resized.channels(), resized.rows, resized.cols};

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());

    std::vector<const char*> inputNames = {inputNamesPtr_.front().get()};
    std::vector<const char*> outputNames = {outputNamesPtr_.front().get()};
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outCount = std::accumulate(outShape.begin(), outShape.end(), int64_t{1}, std::multiplies<int64_t>());
    float* raw = outputTensors.front().GetTensorMutableData<float>();

    int outHeight = static_cast<int>(outShape[2]);
    int outWidth = static_cast<int>(outShape[3]);
    size_t area = static_cast<size_t>(outHeight) * outWidth;

    std::vector<float> predData(area);
    std::vector<uchar> cbufData(area);
    for (size_t i = 0; i < area && static_cast<int64_t>(i) < outCount; i++) {
        predData[i] = raw[i];
        cbufData[i] = static_cast<uchar>(raw[i] * 255);
    }

    cv::Mat predMat(outHeight, outWidth, CV_32F, predData.data());
    cv::Mat cBufMat(outHeight, outWidth, CV_8UC1, cbufData.data());

    cv::Mat thresholdMat;
    cv::threshold(cBufMat, thresholdMat, boxThresh * 255, 255, cv::THRESH_BINARY);

    cv::Mat dilateMat;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(thresholdMat, dilateMat, kernel);

    return findRsBoxes(predMat, dilateMat, scale, boxScoreThresh, unClipRatio);
}

} // namespace arbo::ocr
