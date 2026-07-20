// tests/test_engine.cpp
#include <doctest/doctest.h>
#include "arboOCR/engine.hpp"

using namespace arbo::ocr;

TEST_CASE("EngineConfig has sane defaults") {
    EngineConfig cfg;
    CHECK(cfg.ocrVersion == "PP-OCRv6");
    CHECK(cfg.modelType == "medium"); // recognizer size; see engine.hpp doc comment
    CHECK(cfg.detBoxThresh == doctest::Approx(0.5));
    CHECK(cfg.detThresh == doctest::Approx(0.3));
    CHECK(cfg.detUnclipRatio == doctest::Approx(1.6));
    CHECK(cfg.detLimitSideLen == 1536);
    CHECK(cfg.useAngleCls == false);
}

TEST_CASE("LinePrediction and PagePrediction default-construct cleanly") {
    LinePrediction lp;
    CHECK(lp.text.empty());
    CHECK(lp.score == doctest::Approx(0.0f));
    PagePrediction pp;
    CHECK(pp.image.empty());
    CHECK(pp.lines.empty());
    CHECK(pp.elapsedMs == doctest::Approx(0.0f));
}
