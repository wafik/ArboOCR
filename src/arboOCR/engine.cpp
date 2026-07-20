#include "arboOCR/engine.hpp"

#include <chrono>
#include <filesystem>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

#include "arboOCR/ocr_utils.hpp"

namespace fs = std::filesystem;

namespace arbo::ocr {

bool detectCuda() {
    try {
        auto providers = Ort::GetAvailableProviders();
        for (auto& p : providers) {
            if (p == "CUDAExecutionProvider") return true;
        }
    } catch (...) {
        // never let provider detection crash engine construction
    }
    return false;
}

bool detectTensorrt() {
    try {
        auto providers = Ort::GetAvailableProviders();
        for (auto& p : providers) {
            if (p == "TensorrtExecutionProvider") return true;
        }
    } catch (...) {}
    return false;
}

namespace {

Polygon cvPointsToPolygon(const std::vector<cv::Point>& box) {
    Polygon poly;
    poly.reserve(box.size());
    for (auto& pt : box) {
        poly.push_back({static_cast<float>(pt.x), static_cast<float>(pt.y)});
    }
    return poly;
}

} // namespace

Engine::Engine(const EngineConfig& config) : config_(config) {
    bool useTensorrt = config.useTensorrt && detectTensorrt();
    bool useCuda = (config.useCuda || useTensorrt) && detectCuda();
    backend_ = useTensorrt ? "tensorrt" : useCuda ? "cuda" : "cpu";

    fs::path modelsDir(config.modelsDir);
    std::string detPath = (modelsDir / (config.ocrVersion + "_det.onnx")).string();
    std::string clsPath = (modelsDir / (config.ocrVersion + "_cls.onnx")).string();
    std::string recPath = (modelsDir / (config.ocrVersion + "_rec_" + config.modelType + ".onnx")).string();
    std::string dictPath = (modelsDir / (config.ocrVersion + "_rec_" + config.modelType + "_dict.txt")).string();

    detector_.loadModel(detPath, useCuda, useTensorrt, config.trtCacheDir);
    if (config.useAngleCls) {
        classifier_.loadModel(clsPath, useCuda, useTensorrt, config.trtCacheDir);
    }
    recognizer_.loadModel(recPath, useCuda, useTensorrt, config.trtCacheDir);

    if (!recognizer_.loadKeysFromModelMetadata()) {
        recognizer_.loadKeysFromFile(dictPath);
    }
}

PagePrediction Engine::recognize(const std::string& imagePath) {
    auto t0 = std::chrono::steady_clock::now();
    fs::path path(imagePath);
    PagePrediction result;
    result.image = path.filename().string();

    cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (src.empty()) {
        auto elapsed = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        result.elapsedMs = elapsed;
        return result;
    }

    try {
        ScaleParam scale = getScaleParam(src, config_.detLimitSideLen);
        auto textBoxes = detector_.getTextBoxes(src, scale, config_.detBoxThresh, config_.detThresh, config_.detUnclipRatio);

        std::vector<cv::Mat> partImages;
        partImages.reserve(textBoxes.size());
        for (auto& box : textBoxes) {
            partImages.push_back(getRotateCropImage(src, box.boxPoint));
        }

        auto angles = classifier_.getAngles(partImages, config_.useAngleCls, /*mostAngle=*/false);
        for (size_t i = 0; i < partImages.size(); i++) {
            if (angles[i].index == 1) {
                partImages[i] = matRotateClockWise180(partImages[i]);
            }
        }

        auto textLines = recognizer_.getTextLines(partImages);

        for (size_t i = 0; i < textBoxes.size(); i++) {
            result.lines.push_back(LinePrediction{
                cvPointsToPolygon(textBoxes[i].boxPoint),
                textLines[i].text,
                textBoxes[i].score,
            });
        }
    } catch (const std::exception&) {
        result.lines.clear();
    }

    auto elapsed = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.elapsedMs = elapsed;
    return result;
}

} // namespace arbo::ocr
