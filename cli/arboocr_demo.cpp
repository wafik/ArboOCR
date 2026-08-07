// cli/arboocr_demo.cpp — minimal arboOCR quickstart: recognize one image.
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <cxxopts.hpp>

#include <opencv2/imgcodecs.hpp>

#include "arboOCR/engine.hpp"
#include "arboOCR/logging.hpp"
#include "arboOCR/markdown.hpp"
#include "arboOCR/visualize.hpp"

int main(int argc, char* argv[]) {
    // Library default is silent (no callback). --log-level opts in below.
    cxxopts::Options opts("arboocr_demo", "Recognize text in one image with arboOCR");
    opts.add_options()
        ("image", "Path to input image", cxxopts::value<std::string>())
        ("models-dir", "Directory containing ONNX models", cxxopts::value<std::string>()->default_value("models"))
        ("ocr-version", "OCR model version", cxxopts::value<std::string>()->default_value("PP-OCRv6"))
        ("model-type", "Recognizer size — tiny/small/medium (default small; detector is always the same file unless --det-model)",
            cxxopts::value<std::string>()->default_value("small"))
        // Detection tuning (accuracy knobs — see EngineConfig in engine.hpp).
        ("det-limit-side-len", "Longest image side for detection resize",
            cxxopts::value<int>()->default_value("960"))
        ("det-thresh", "Detection probability-map binarization threshold",
            cxxopts::value<float>()->default_value("0.3"))
        ("det-box-thresh", "Minimum mean score for a detected box to be kept",
            cxxopts::value<float>()->default_value("0.5"))
        ("det-unclip-ratio", "Box expansion ratio applied after binarization",
            cxxopts::value<float>()->default_value("1.6"))
        ("split-overmerged", "Split wide det boxes that fuse two side-by-side fields (ink-gap heuristic)",
            cxxopts::value<bool>()->default_value("false"))
        // Recognition tuning.
        ("rec-batch-num", "Crops per recognition inference call (clamped to >= 1)",
            cxxopts::value<int>()->default_value("6"))
        ("min-confidence", "Drop lines below this recognition confidence (0 disables filtering)",
            cxxopts::value<float>()->default_value("0.5"))
        ("angle", "Enable angle classification", cxxopts::value<bool>()->default_value("false"))
        ("cuda", "Request CUDA execution provider", cxxopts::value<bool>()->default_value("false"))
        ("tensorrt", "Request TensorRT execution provider", cxxopts::value<bool>()->default_value("false"))
        ("fp16", "TensorRT FP16 (default true; only used with --tensorrt)",
            cxxopts::value<bool>()->default_value("true"))
        ("trt-cache-dir", "Directory for cached TensorRT engines (only used with --tensorrt)",
            cxxopts::value<std::string>()->default_value("models/trt_engines"))
        ("clahe", "Apply CLAHE contrast enhancement before detection (helps low-contrast/faded documents)",
            cxxopts::value<bool>()->default_value("false"))
        ("json", "Print machine-readable JSON (only JSON on stdout; suppresses the human-readable lines)",
            cxxopts::value<bool>()->default_value("false"))
        ("log-level", "Log engine events to stderr at this level — debug|info|warn|error (default: silent)",
            cxxopts::value<std::string>())
        ("draw", "Write a copy of the image with detected boxes outlined to this path",
            cxxopts::value<std::string>())
        ("markdown", "Write the reconstructed markdown document to this path (implies --word-boxes)",
            cxxopts::value<std::string>())
        ("word-boxes", "Also emit a polygon per word (per character for CJK)",
            cxxopts::value<bool>()->default_value("false"))
        ("det-model", "Override detector ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("cls-model", "Override classifier ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("rec-model", "Override recognizer ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("dict", "Override character dict path (fallback if no ONNX metadata)",
            cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = opts.parse(argc, argv);
    } catch (const std::exception& e) {
        // Uncaught, this exception unwinds past main() and hits
        // std::terminate() — which this toolchain's hardened runtime turns
        // into an unhelpful STATUS_STACK_BUFFER_OVERRUN crash instead of a
        // clean error (e.g. any unrecognized flag). Fail cleanly instead.
        std::cerr << "arboocr_demo: " << e.what() << "\n\n" << opts.help() << std::endl;
        return 1;
    }
    if (result.count("help") || !result.count("image")) {
        std::cout << opts.help() << std::endl;
        return result.count("image") ? 0 : 1;
    }

    // Opt-in logging: without --log-level the library stays silent (no callback).
    if (result.count("log-level")) {
        const std::string level = result["log-level"].as<std::string>();
        arbo::ocr::LogLevel minLevel = arbo::ocr::LogLevel::Info;
        if (level == "debug") {
            minLevel = arbo::ocr::LogLevel::Debug;
        } else if (level == "info") {
            minLevel = arbo::ocr::LogLevel::Info;
        } else if (level == "warn") {
            minLevel = arbo::ocr::LogLevel::Warn;
        } else if (level == "error") {
            minLevel = arbo::ocr::LogLevel::Error;
        } else {
            std::cerr << "arboocr_demo: unknown --log-level \"" << level
                      << "\" (expected one of: debug, info, warn, error)" << std::endl;
            return 1;
        }
        arbo::ocr::setLogCallback(arbo::ocr::makeStderrLogger());
        arbo::ocr::setMinLogLevel(minLevel);
    }

    arbo::ocr::EngineConfig cfg;
    cfg.modelsDir = result["models-dir"].as<std::string>();
    cfg.ocrVersion = result["ocr-version"].as<std::string>();
    cfg.modelType = result["model-type"].as<std::string>();
    cfg.detLimitSideLen = result["det-limit-side-len"].as<int>();
    cfg.detThresh = result["det-thresh"].as<float>();
    cfg.detBoxThresh = result["det-box-thresh"].as<float>();
    cfg.detUnclipRatio = result["det-unclip-ratio"].as<float>();
    cfg.splitOvermerged = result["split-overmerged"].as<bool>();
    cfg.recBatchNum = result["rec-batch-num"].as<int>();
    cfg.minimumConfidence = result["min-confidence"].as<float>();
    cfg.useAngleCls = result["angle"].as<bool>();
    cfg.useCuda = result["cuda"].as<bool>();
    cfg.useTensorrt = result["tensorrt"].as<bool>();
    cfg.useFp16 = result["fp16"].as<bool>();
    cfg.trtCacheDir = result["trt-cache-dir"].as<std::string>();
    cfg.useClahe = result["clahe"].as<bool>();
    // --markdown detects `key | value` rows from the wide gap between words, so
    // it needs word boxes even when --word-boxes was not asked for.
    cfg.returnWordBoxes = result["word-boxes"].as<bool>() || result.count("markdown") > 0;
    cfg.detModelPath = result["det-model"].as<std::string>();
    cfg.clsModelPath = result["cls-model"].as<std::string>();
    cfg.recModelPath = result["rec-model"].as<std::string>();
    cfg.dictPath = result["dict"].as<std::string>();

    const bool jsonMode = result["json"].as<bool>();

    // Engine construction loads the ONNX sessions and throws Ort::Exception on
    // a missing/corrupt model. Uncaught, that unwinds past main() the same way
    // a cxxopts parse error would — report the resolved paths and bail cleanly.
    std::unique_ptr<arbo::ocr::Engine> engine;
    try {
        engine = std::make_unique<arbo::ocr::Engine>(cfg);
    } catch (const std::exception& e) {
        const arbo::ocr::ModelPaths paths = arbo::ocr::resolveModelPaths(cfg);
        std::cerr << "arboocr_demo: failed to load models: " << e.what() << "\n"
                  << "  det:  " << paths.det << "\n"
                  << "  cls:  " << paths.cls << "\n"
                  << "  rec:  " << paths.rec << "\n"
                  << "  dict: " << paths.dict << "\n"
                  << "Check --models-dir/--ocr-version/--model-type, or override the"
                     " paths with --det-model/--cls-model/--rec-model/--dict."
                  << std::endl;
        return 2;
    }

    if (!jsonMode) {
        std::cout << "Backend: " << engine->backend() << "\n";
    }

    // recognize() is documented as never throwing; guard anyway so an
    // unexpected inference failure is a diagnostic, not a terminate().
    arbo::ocr::PagePrediction page;
    try {
        page = engine->recognize(result["image"].as<std::string>());
    } catch (const std::exception& e) {
        std::cerr << "arboocr_demo: recognition failed: " << e.what() << std::endl;
        return 2;
    }

    // Before the jsonMode early-return, so --draw composes with --json.
    // Diagnostics go to stderr to keep stdout pure JSON in that mode.
    if (result.count("draw")) {
        const auto& outPath = result["draw"].as<std::string>();
        cv::Mat src = cv::imread(result["image"].as<std::string>(), cv::IMREAD_COLOR);
        if (src.empty()) {
            std::cerr << "arboocr_demo: --draw: cannot re-read image for overlay\n";
        } else if (!cv::imwrite(outPath, arbo::ocr::drawResult(src, page))) {
            std::cerr << "arboocr_demo: --draw: failed to write " << outPath << "\n";
        } else if (!jsonMode) {
            std::cout << "Overlay: " << outPath << "\n";
        }
    }

    // Same placement as --draw, for the same reason: composes with --json, and
    // a write failure is a warning because the recognition itself succeeded.
    if (result.count("markdown")) {
        const auto& outPath = result["markdown"].as<std::string>();
        std::ofstream out(outPath, std::ios::binary);
        if (!out) {
            std::cerr << "arboocr_demo: --markdown: cannot open " << outPath << "\n";
        } else {
            out << arbo::ocr::toMarkdown(page);
            out.close();
            if (!out) {
                std::cerr << "arboocr_demo: --markdown: failed to write " << outPath << "\n";
            } else if (!jsonMode) {
                std::cout << "Markdown: " << outPath << "\n";
            }
        }
    }

    if (jsonMode) {
        // --markdown turns word boxes on internally, but the JSON payload is a
        // published contract: "words" appears iff the caller asked for it. Drop
        // them again so --markdown can't silently reshape a wrapper's output.
        if (!result["word-boxes"].as<bool>()) {
            for (auto& line : page.lines) line.words.clear();
        }
        // Pure JSON on stdout, nothing else — callers (e.g. the PHP wrapper)
        // json_decode() the whole stream. Empty lines is still success.
        std::cout << arbo::ocr::toJson(page, engine->backend()) << "\n";
        return 0;
    }

    std::cout << "Image: " << page.image << "\n";
    // recognize() never throws — missing/unreadable images and inference
    // failures both yield empty lines with elapsedMs still set.
    if (page.lines.empty()) {
        std::cerr << "No text found (missing image, unsupported format, or empty page)"
                  << " (" << page.elapsedMs << " ms)\n";
        return 1;
    }
    std::cout << "Lines: " << page.lines.size() << " (" << page.elapsedMs << " ms)\n";
    for (size_t i = 0; i < page.lines.size(); i++) {
        const auto& line = page.lines[i];
        std::cout << "  [" << i << "] \"" << line.text
                  << "\" (score=" << line.score
                  << " detScore=" << line.detScore << ")";
        if (!line.polygon.empty()) {
            std::cout << " poly=[";
            for (size_t p = 0; p < line.polygon.size(); p++) {
                if (p) std::cout << ", ";
                std::cout << "(" << line.polygon[p].x << "," << line.polygon[p].y << ")";
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
    return 0;
}
