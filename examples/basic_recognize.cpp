// examples/basic_recognize.cpp
//
// The most common way to use arboOCR: construct an Engine, call
// recognize() on an image, print the lines it found (text, score, and
// bounding polygon). This is the pattern shown in the README's Quickstart
// section — this file is that snippet, made buildable so it's provably
// correct rather than just illustrative.
//
// Usage:
//   basic_recognize <image> [models-dir] [--tensorrt]
//
// Defaults to CPU: TensorRT's first run on a new model compiles an engine
// from scratch, which can take several minutes even for a modest model on
// weaker hardware (e.g. a Jetson Nano) — a bad first impression for a demo
// meant to run instantly. Pass --tensorrt to opt in once you're ready for
// that one-time cost (subsequent runs reuse the cached engine and are
// fast — see EngineConfig::trtCacheDir).
//
#include <arboOCR/engine.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image> [models-dir] [--tensorrt]\n";
        return 1;
    }
    std::string imagePath = argv[1];
    std::string modelsDir = "models";
    bool useTensorrt = false;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tensorrt") useTensorrt = true;
        else modelsDir = arg;
    }

    arbo::ocr::EngineConfig cfg;
    cfg.modelsDir = modelsDir;
    cfg.useTensorrt = useTensorrt; // auto-falls back to CUDA, then CPU — see engine.backend()

    arbo::ocr::Engine engine(cfg);
    std::cout << "Backend: " << engine.backend() << "\n";

    arbo::ocr::PagePrediction page = engine.recognize(imagePath);
    // recognize() never throws: missing/unreadable images and inference
    // failures both return empty lines with elapsedMs still set.
    if (page.lines.empty()) {
        std::cerr << "No text found (missing image, unsupported format, or empty page)"
                  << " in " << page.elapsedMs << " ms\n";
        return 1;
    }

    std::cout << "Found " << page.lines.size() << " lines in " << page.elapsedMs << " ms\n\n";

    for (const auto& line : page.lines) {
        std::cout << line.text << "  (score=" << line.score << ")\n";
        // Polygon: 4 points, clockwise from top-left-ish — use these to draw
        // boxes over the source image (e.g. cv::polylines).
        std::cout << "  polygon:";
        for (const auto& pt : line.polygon) {
            std::cout << " (" << pt.x << "," << pt.y << ")";
        }
        std::cout << "\n";
    }

    return 0;
}
