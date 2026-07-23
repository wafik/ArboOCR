#include "arboOCR/engine.hpp"

#include <chrono>
#include <filesystem>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

#include "arboOCR/logging.hpp"
#include "arboOCR/ocr_utils.hpp"
#include "arboOCR/preprocess.hpp"

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

ModelPaths resolveModelPaths(const EngineConfig& cfg) {
    fs::path modelsDir(cfg.modelsDir);
    ModelPaths out;
    out.det = !cfg.detModelPath.empty()
        ? cfg.detModelPath
        : (modelsDir / (cfg.ocrVersion + "_det.onnx")).string();
    out.cls = !cfg.clsModelPath.empty()
        ? cfg.clsModelPath
        : (modelsDir / (cfg.ocrVersion + "_cls.onnx")).string();
    out.rec = !cfg.recModelPath.empty()
        ? cfg.recModelPath
        : (modelsDir / (cfg.ocrVersion + "_rec_" + cfg.modelType + ".onnx")).string();
    out.dict = !cfg.dictPath.empty()
        ? cfg.dictPath
        : (modelsDir / (cfg.ocrVersion + "_rec_" + cfg.modelType + "_dict.txt")).string();
    return out;
}

Engine::Engine(const EngineConfig& config) : config_(config) {
    bool useTensorrt = config.useTensorrt && detectTensorrt();
    bool useCuda = (config.useCuda || useTensorrt) && detectCuda();
    backend_ = useTensorrt ? "tensorrt" : useCuda ? "cuda" : "cpu";
    log(LogLevel::Info, "Engine backend: " + backend_
        + (useTensorrt ? (config.useFp16 ? " (fp16)" : " (fp32)") : ""));

    ModelPaths paths = resolveModelPaths(config);
    log(LogLevel::Debug,
        "Model paths det=" + paths.det + " cls=" + paths.cls
        + " rec=" + paths.rec + " dict=" + paths.dict);

    // Must be set before loadModel so TensorRT profiles match runtime batch size.
    recognizer_.setRecBatchNum(config.recBatchNum);

    detector_.loadModel(paths.det, useCuda, useTensorrt, config.trtCacheDir, config.useFp16);
    if (config.useAngleCls) {
        classifier_.loadModel(paths.cls, useCuda, useTensorrt, config.trtCacheDir, config.useFp16);
    }
    recognizer_.loadModel(paths.rec, useCuda, useTensorrt, config.trtCacheDir, config.useFp16);

    if (!recognizer_.loadKeysFromModelMetadata()) {
        log(LogLevel::Debug, "Recognizer keys: model metadata missing, loading " + paths.dict);
        recognizer_.loadKeysFromFile(paths.dict);
    } else {
        log(LogLevel::Debug, "Recognizer keys: loaded from model metadata");
    }
    if (recognizer_.keyCount() == 0) {
        log(LogLevel::Warn, "Recognizer has no character dictionary loaded");
    }
}

PagePrediction Engine::runPipeline(const cv::Mat& src, const std::string& imageName) {
    auto t0 = std::chrono::steady_clock::now();
    PagePrediction result;
    result.image = imageName;

    if (src.empty()) {
        log(LogLevel::Warn, imageName.empty()
            ? "recognize: empty image buffer"
            : "recognize: empty/unreadable image (" + imageName + ")");
        auto elapsed = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        result.elapsedMs = elapsed;
        return result;
    }

    try {
        cv::Mat prepped = config_.useClahe ? applyClahe(src) : src;
        ScaleParam scale = getScaleParam(prepped, config_.detLimitSideLen);
        auto textBoxes = detector_.getTextBoxes(prepped, scale, config_.detBoxThresh, config_.detThresh, config_.detUnclipRatio);
        if (config_.splitOvermerged) {
            textBoxes = expandOvermergedBoxes(textBoxes, prepped);
        }

        std::vector<cv::Mat> partImages;
        partImages.reserve(textBoxes.size());
        for (auto& box : textBoxes) {
            partImages.push_back(getRotateCropImage(prepped, box.boxPoint));
        }

        auto angles = classifier_.getAngles(partImages, config_.useAngleCls, /*mostAngle=*/false);
        for (size_t i = 0; i < partImages.size(); i++) {
            if (angles[i].index == 1) {
                partImages[i] = matRotateClockWise180(partImages[i]);
            }
        }

        auto textLines = recognizer_.getTextLines(partImages);

        for (size_t i = 0; i < textBoxes.size(); i++) {
            const float recScore = (i < textLines.size())
                ? meanRecScore(textLines[i].charScores)
                : 0.0f;
            const std::string& text = (i < textLines.size()) ? textLines[i].text : std::string{};
            result.lines.push_back(LinePrediction{
                cvPointsToPolygon(textBoxes[i].boxPoint),
                text,
                recScore,
                textBoxes[i].score,
            });
        }
        sortLinesReadingOrder(result.lines);
        log(LogLevel::Debug, "recognize: " + std::to_string(result.lines.size()) + " lines");
    } catch (const std::exception& ex) {
        log(LogLevel::Error, std::string("recognize failed: ") + ex.what());
        result.lines.clear();
    }

    auto elapsed = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.elapsedMs = elapsed;
    return result;
}

PagePrediction Engine::recognize(const std::string& imagePath) {
    fs::path path(imagePath);
    cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (src.empty()) {
        log(LogLevel::Warn, "recognize: failed to read image path: " + imagePath);
    }
    return runPipeline(src, path.filename().string());
}

PagePrediction Engine::recognize(const cv::Mat& image) {
    return runPipeline(image, {});
}

std::future<PagePrediction> Engine::recognizeAsync(const std::string& imagePath) {
    return std::async(std::launch::async, [this, imagePath]() {
        return recognize(imagePath);
    });
}

std::future<PagePrediction> Engine::recognizeAsync(const cv::Mat& image) {
    cv::Mat copy = image.clone();
    return std::async(std::launch::async, [this, copy = std::move(copy)]() {
        return recognize(copy);
    });
}

} // namespace arbo::ocr
