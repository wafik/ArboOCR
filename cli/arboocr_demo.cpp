// cli/arboocr_demo.cpp — minimal arboOCR quickstart: recognize one image.
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "arboOCR/engine.hpp"
#include "arboOCR/logging.hpp"

int main(int argc, char* argv[]) {
    // Demo only: library default is silent. Route engine events to stderr.
    arbo::ocr::setLogCallback(arbo::ocr::makeStderrLogger());
    cxxopts::Options opts("arboocr_demo", "Recognize text in one image with arboOCR");
    opts.add_options()
        ("image", "Path to input image", cxxopts::value<std::string>())
        ("models-dir", "Directory containing ONNX models", cxxopts::value<std::string>()->default_value("models"))
        ("ocr-version", "OCR model version", cxxopts::value<std::string>()->default_value("PP-OCRv6"))
        ("model-type", "Recognizer size — tiny/small/medium (detector is always the same file; medium is most accurate, tiny is fastest)",
            cxxopts::value<std::string>()->default_value("medium"))
        ("angle", "Enable angle classification", cxxopts::value<bool>()->default_value("false"))
        ("cuda", "Request CUDA execution provider", cxxopts::value<bool>()->default_value("false"))
        ("tensorrt", "Request TensorRT execution provider", cxxopts::value<bool>()->default_value("false"))
        ("fp16", "TensorRT FP16 (default true; only used with --tensorrt)",
            cxxopts::value<bool>()->default_value("true"))
        ("h,help", "Print usage");

    auto result = opts.parse(argc, argv);
    if (result.count("help") || !result.count("image")) {
        std::cout << opts.help() << std::endl;
        return result.count("image") ? 0 : 1;
    }

    arbo::ocr::EngineConfig cfg;
    cfg.modelsDir = result["models-dir"].as<std::string>();
    cfg.ocrVersion = result["ocr-version"].as<std::string>();
    cfg.modelType = result["model-type"].as<std::string>();
    cfg.useAngleCls = result["angle"].as<bool>();
    cfg.useCuda = result["cuda"].as<bool>();
    cfg.useTensorrt = result["tensorrt"].as<bool>();
    cfg.useFp16 = result["fp16"].as<bool>();

    arbo::ocr::Engine engine(cfg);
    std::cout << "Backend: " << engine.backend() << "\n";

    auto page = engine.recognize(result["image"].as<std::string>());
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
                  << "\" (score=" << line.score << ")";
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
