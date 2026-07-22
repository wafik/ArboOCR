// tests/test_engine.cpp
#include <doctest/doctest.h>

#include "arboOCR/engine.hpp"
#include "arboOCR/recognizer.hpp"
#include "arboOCR/types.hpp"

using namespace arbo::ocr;

TEST_CASE("EngineConfig has sane defaults") {
    EngineConfig cfg;
    CHECK(cfg.ocrVersion == "PP-OCRv6");
    CHECK(cfg.modelType == "medium"); // recognizer size; see engine.hpp doc comment
    CHECK(cfg.detBoxThresh == doctest::Approx(0.5));
    CHECK(cfg.detThresh == doctest::Approx(0.3));
    CHECK(cfg.detUnclipRatio == doctest::Approx(1.6));
    CHECK(cfg.detLimitSideLen == 1536);
    CHECK(cfg.recBatchNum == 6);
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

TEST_CASE("toJson serializes page with polygon, score, and escaped text") {
    PagePrediction page;
    page.image = "a\"b\\c";
    page.elapsedMs = 12.5f;
    page.lines.push_back(LinePrediction{
        {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}},
        "hello\nworld",
        0.91f,
    });

    const std::string json = toJson(page);
    CHECK(json.find("\"image\":\"a\\\"b\\\\c\"") != std::string::npos);
    CHECK(json.find("\"elapsedMs\":") != std::string::npos);
    CHECK(json.find("\"text\":\"hello\\nworld\"") != std::string::npos);
    CHECK(json.find("\"score\":") != std::string::npos);
    CHECK(json.find("\"x\":1") != std::string::npos);
    CHECK(json.find("\"y\":2") != std::string::npos);
    CHECK(json.find("\"polygon\":[") != std::string::npos);

    const std::string pretty = toJson(page, true);
    CHECK(pretty.find('\n') != std::string::npos);
    CHECK(pretty.find("\"lines\":[") != std::string::npos);
}

TEST_CASE("toJson on empty page is valid object") {
    PagePrediction empty;
    empty.image = "none.jpg";
    const std::string json = toJson(empty);
    CHECK(json == "{\"image\":\"none.jpg\",\"elapsedMs\":0,\"lines\":[]}");
}

TEST_CASE("Recognizer setRecBatchNum clamps and reports") {
    Recognizer rec;
    CHECK(rec.recBatchNum() == 6);
    rec.setRecBatchNum(16);
    CHECK(rec.recBatchNum() == 16);
    rec.setRecBatchNum(0);
    CHECK(rec.recBatchNum() == 1);
    rec.setRecBatchNum(-3);
    CHECK(rec.recBatchNum() == 1);
}
