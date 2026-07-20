// cli/arboocr_demo.cpp — minimal arboOCR quickstart: recognize one image.
#include <iostream>
#include <string>

#include <cxxopts.hpp>

#include "arboOCR/engine.hpp"

int main(int argc, char* argv[]) {
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

    arbo::ocr::Engine engine(cfg);
    std::cout << "Backend: " << engine.backend() << "\n";

    auto page = engine.recognize(result["image"].as<std::string>());
    std::cout << "Image: " << page.image << "\n";
    std::cout << "Lines: " << page.lines.size() << " (" << page.elapsedMs << " ms)\n";
    for (size_t i = 0; i < page.lines.size(); i++) {
        std::cout << "  [" << i << "] \"" << page.lines[i].text
                  << "\" (score=" << page.lines[i].score << ")\n";
    }
    return 0;
}
