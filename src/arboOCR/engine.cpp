#include "arboOCR/engine.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

#include "arboOCR/logging.hpp"
#include "arboOCR/model_downloader.hpp"
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

cv::Mat decodeImageBytes(const uint8_t* data, size_t size) {
    // cv::imdecode CV_Asserts (throws) on an empty buffer instead of returning
    // an empty Mat, and the Mat header below takes an int column count — so
    // null/empty/oversized inputs are screened here rather than by imdecode.
    if (data == nullptr || size == 0
        || size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cv::Mat();
    }
    try {
        // Non-owning 1xN byte view over the caller's buffer: no copy, and
        // imdecode only reads it. const_cast is safe for the same reason.
        const cv::Mat buf(1, static_cast<int>(size), CV_8UC1,
                          const_cast<uint8_t*>(data));
        return cv::imdecode(buf, cv::IMREAD_COLOR);
    } catch (const std::exception&) {
        // Truncated/malformed payloads can throw out of a codec rather than
        // returning empty. Same degradation either way.
        return cv::Mat();
    }
}

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

bool modelDownloadsAllowed(const EngineConfig& cfg) {
    if (!cfg.autoDownload) return false;
    const char* offline = std::getenv("ARBOOCR_OFFLINE");
    // Set-and-non-empty and not "0" engages offline mode. Unset, empty, and
    // "0" all mean "not engaged" — "0" especially, so a wrapper script that
    // exports ARBOOCR_OFFLINE=0 to *disable* offline mode is believed.
    return !(offline && *offline && std::string(offline) != "0");
}

ModelPaths ensureOcrModels(const EngineConfig& cfg) {
    ModelPaths paths = resolveModelPaths(cfg);

    if (!modelDownloadsAllowed(cfg)) {
        return paths;
    }

    // Same order as ocrModelFileNames(): det, cls, rec, dict.
    const std::vector<std::string> names = ocrModelFileNames(cfg.ocrVersion, cfg.modelType);
    std::string* const slots[4] = {&paths.det, &paths.cls, &paths.rec, &paths.dict};
    const bool wanted[4] = {true, cfg.useAngleCls, true, true};
    const bool overridden[4] = {
        !cfg.detModelPath.empty(), !cfg.clsModelPath.empty(),
        !cfg.recModelPath.empty(), !cfg.dictPath.empty()};

    const fs::path cacheDir(defaultModelsCacheDir());
    std::string baseUrl = cfg.modelsBaseUrl.empty() ? defaultModelsBaseUrl() : cfg.modelsBaseUrl;
    if (!baseUrl.empty() && baseUrl.back() != '/') baseUrl.push_back('/');

    for (int i = 0; i < 4; ++i) {
        if (!wanted[i] || overridden[i]) continue;
        std::error_code ec;
        if (fs::exists(*slots[i], ec) && fs::file_size(*slots[i], ec) > 0) continue;

        const std::string dest = (cacheDir / names[i]).string();
        // downloadFile re-hashes an existing destination rather than trusting
        // it, so a cache hit still costs a read. That is deliberate — a file
        // that rotted on disk after it was written would otherwise be mmap'd
        // straight into ONNX Runtime — and it is cheap next to building the
        // ORT session we are about to build anyway.
        const bool cached = fs::exists(dest, ec) && fs::file_size(dest, ec) > 0;
        log(LogLevel::Info, (cached ? "Verifying cached " : "Fetching ")
            + names[i] + " -> " + dest);
        const DownloadResult r = downloadFile(baseUrl + names[i], dest, knownSha256(names[i]));
        if (r.ok) {
            *slots[i] = dest;
        } else {
            // The dict is routinely absent by design; the rest is a real
            // problem, but loadModel reports it better than we can here.
            log(i == 3 ? LogLevel::Debug : LogLevel::Warn,
                "Could not fetch " + names[i] + ": " + r.errorMessage);
        }
    }
    return paths;
}

Engine::Engine(const EngineConfig& config) : config_(config) {
    bool useTensorrt = config.useTensorrt && detectTensorrt();
    bool useCuda = (config.useCuda || useTensorrt) && detectCuda();
    backend_ = useTensorrt ? "tensorrt" : useCuda ? "cuda" : "cpu";
    log(LogLevel::Info, "Engine backend: " + backend_
        + (useTensorrt ? (config.useFp16 ? " (fp16)" : " (fp32)") : ""));

    ModelPaths paths = ensureOcrModels(config);
    log(LogLevel::Debug,
        "Model paths det=" + paths.det + " cls=" + paths.cls
        + " rec=" + paths.rec + " dict=" + paths.dict);

    // Must be set before loadModel so TensorRT profiles match runtime batch size.
    recognizer_.setRecBatchNum(config.recBatchNum);
    recognizer_.setReturnSpans(config.returnWordBoxes);

    detector_.loadModel(paths.det, useCuda, useTensorrt, config.trtCacheDir, config.useFp16,
        config.intraOpNumThreads, config.interOpNumThreads, config.enableCpuMemArena);
    if (config.useAngleCls) {
        classifier_.loadModel(paths.cls, useCuda, useTensorrt, config.trtCacheDir, config.useFp16,
            config.intraOpNumThreads, config.interOpNumThreads, config.enableCpuMemArena);
    }
    recognizer_.loadModel(paths.rec, useCuda, useTensorrt, config.trtCacheDir, config.useFp16,
        config.intraOpNumThreads, config.interOpNumThreads, config.enableCpuMemArena);

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

    // ponytail: stage timings go to --log-level debug only; no struct/JSON
    // change, zero overhead when silent (chrono reads only).
    float detMs = 0.0f, cropMs = 0.0f, clsMs = 0.0f, recMs = 0.0f;
    try {
        cv::Mat prepped = config_.useClahe ? applyClahe(src) : src;
        ScaleParam scale = getScaleParam(prepped, config_.detLimitSideLen);
        const auto tDet0 = std::chrono::steady_clock::now();
        auto textBoxes = detector_.getTextBoxes(prepped, scale, config_.detBoxThresh, config_.detThresh, config_.detUnclipRatio);
        detMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tDet0).count();
        if (config_.splitOvermerged) {
            textBoxes = expandOvermergedBoxes(textBoxes, prepped);
        }

        std::vector<cv::Mat> partImages;
        partImages.reserve(textBoxes.size());
        // Word boxes need to undo whatever getRotateCropImage/angle-cls did to
        // the crop before a horizontal fraction of it means anything on the
        // page: a tall box is read along the quad's vertical axis, and a
        // 180-flipped crop is read backwards. Recording per box beats
        // recomputing the conditions later, which would drift.
        std::vector<char> wasTransposed(textBoxes.size(), 0);
        std::vector<char> wasRotated180(textBoxes.size(), 0);
        const auto tCrop0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < textBoxes.size(); i++) {
            bool transposed = false;
            partImages.push_back(getRotateCropImage(prepped, textBoxes[i].boxPoint, &transposed));
            wasTransposed[i] = transposed ? 1 : 0;
        }
        cropMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tCrop0).count();

        const auto tCls0 = std::chrono::steady_clock::now();
        auto angles = classifier_.getAngles(partImages, config_.useAngleCls, /*mostAngle=*/false);
        for (size_t i = 0; i < partImages.size(); i++) {
            if (angles[i].index == 1) {
                partImages[i] = matRotateClockWise180(partImages[i]);
                wasRotated180[i] = 1;
            }
        }
        clsMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tCls0).count();

        const auto tRec0 = std::chrono::steady_clock::now();
        auto textLines = recognizer_.getTextLines(partImages);
        recMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tRec0).count();

        for (size_t i = 0; i < textBoxes.size(); i++) {
            const float recScore = (i < textLines.size())
                ? meanRecScore(textLines[i].charScores)
                : 0.0f;
            const std::string& text = (i < textLines.size()) ? textLines[i].text : std::string{};
            if (!keepByConfidence(text, recScore, config_.minimumConfidence)) {
                continue;
            }
            LinePrediction line{
                cvPointsToPolygon(textBoxes[i].boxPoint),
                text,
                recScore,
                textBoxes[i].score,
                {},
            };
            if (config_.returnWordBoxes && i < textLines.size()) {
                line.words = groupTokensIntoWords(
                    textLines[i].tokens, textLines[i].spans, textLines[i].charScores,
                    line.polygon, wasTransposed[i] != 0, wasRotated180[i] != 0);
                // The tokens predate refineDecodedText, which runs on the joined
                // string and has no index map back. Applying the same rules per
                // word keeps word text consistent with line text; splitting on
                // spaces already absorbed the space-collapsing half.
                for (auto& w : line.words) refineDecodedText(w.text);
            }
            result.lines.push_back(std::move(line));
        }
        sortLinesReadingOrder(result.lines);
        log(LogLevel::Debug, "recognize: " + std::to_string(result.lines.size()) + " lines");
        // Sum of stage spans can slightly exceed elapsedMs (nesting-free here,
        // but chrono reads are not atomic with inference) — treat as budget
        // shares, not an exact partition.
        log(LogLevel::Debug, "stages: det=" + std::to_string(detMs)
            + "ms crop=" + std::to_string(cropMs)
            + "ms cls=" + std::to_string(clsMs)
            + "ms rec=" + std::to_string(recMs)
            + "ms boxes=" + std::to_string(textBoxes.size()));
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

PagePrediction Engine::recognizeEncoded(const uint8_t* data, size_t size) {
    cv::Mat src = decodeImageBytes(data, size);
    if (src.empty()) {
        log(LogLevel::Warn, "recognize: failed to decode "
            + std::to_string(size) + " encoded bytes");
    }
    return runPipeline(src, {});
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

std::future<PagePrediction> Engine::recognizeEncodedAsync(const uint8_t* data, size_t size) {
    // Copy rather than capture the pointer: a raw pointer outliving its buffer
    // is the classic async footgun, and encoded bytes are smaller than the mat
    // the sync path decodes anyway — the Mat overload already clones for this
    // exact reason. Guarded so a null pointer never forms an invalid range.
    std::vector<uint8_t> buf;
    if (data != nullptr && size > 0) {
        buf.assign(data, data + size);
    }
    return std::async(std::launch::async, [this, buf = std::move(buf)]() {
        return recognizeEncoded(buf.data(), buf.size());
    });
}

} // namespace arbo::ocr
