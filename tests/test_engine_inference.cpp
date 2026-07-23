#include <doctest/doctest.h>
#include <iostream>
#include "arboOCR/engine.hpp"

using namespace arbo::ocr;

TEST_CASE("Engine loads tiny models and runs recognize on a test image"
          * doctest::skip(true)) { // un-skip once models/ is populated locally
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.modelType = "tiny";
    cfg.useAngleCls = false;

    Engine engine(cfg);
    CHECK(engine.backend() == "cpu");

    auto result = engine.recognize("tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg");
    CHECK_FALSE(result.image.empty());
    CHECK(result.elapsedMs > 0.0f);

    std::cout << "Image: " << result.image << "\n";
    std::cout << "Lines detected: " << result.lines.size() << "\n";
    std::cout << "Elapsed: " << result.elapsedMs << " ms\n";
    for (size_t i = 0; i < result.lines.size(); i++) {
        std::cout << "  Line " << i << ": \"" << result.lines[i].text
                  << "\" (score=" << result.lines[i].score << ")\n";
    }
}

TEST_CASE("Engine with useClahe=true runs recognize without crashing"
          * doctest::skip(true)) { // un-skip once models/ is populated locally
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.modelType = "tiny";
    cfg.useAngleCls = false;
    cfg.useClahe = true;

    Engine engine(cfg);
    CHECK(engine.backend() == "cpu");

    auto result = engine.recognize("tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg");
    CHECK_FALSE(result.image.empty());
    CHECK(result.elapsedMs > 0.0f);

    std::cout << "useClahe=true — Lines detected: " << result.lines.size() << "\n";
}
