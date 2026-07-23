// tests/test_engine.cpp
#include <doctest/doctest.h>

#include <filesystem>

#include "arboOCR/engine.hpp"
#include "arboOCR/recognizer.hpp"
#include "arboOCR/types.hpp"

using namespace arbo::ocr;

TEST_CASE("EngineConfig has sane defaults") {
    EngineConfig cfg;
    CHECK(cfg.ocrVersion == "PP-OCRv6");
    CHECK(cfg.modelType == "small"); // was medium; CPU-honest default
    CHECK(cfg.detBoxThresh == doctest::Approx(0.5));
    CHECK(cfg.detThresh == doctest::Approx(0.3));
    CHECK(cfg.detUnclipRatio == doctest::Approx(1.6));
    CHECK(cfg.detLimitSideLen == 960);
    CHECK(cfg.recBatchNum == 6);
    CHECK(cfg.useAngleCls == false);
    CHECK(cfg.useFp16 == true); // TensorRT FP16 default (was always-on)
    CHECK(cfg.useClahe == false);
    CHECK(cfg.splitOvermerged == false);
    CHECK(cfg.detModelPath.empty());
    CHECK(cfg.clsModelPath.empty());
    CHECK(cfg.recModelPath.empty());
    CHECK(cfg.dictPath.empty());
}

TEST_CASE("EngineConfig path overrides default empty") {
    EngineConfig cfg;
    CHECK(cfg.detModelPath.empty());
    CHECK(cfg.clsModelPath.empty());
    CHECK(cfg.recModelPath.empty());
    CHECK(cfg.dictPath.empty());
}

TEST_CASE("resolveModelPaths uses default flat layout under modelsDir") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.ocrVersion = "PP-OCRv6";
    cfg.modelType = "medium";
    ModelPaths p = resolveModelPaths(cfg);
    // Use path-agnostic checks: ends with expected filename; prefer
    // std::filesystem for separator safety on Windows.
    namespace fs = std::filesystem;
    CHECK(fs::path(p.det).filename() == "PP-OCRv6_det.onnx");
    CHECK(fs::path(p.cls).filename() == "PP-OCRv6_cls.onnx");
    CHECK(fs::path(p.rec).filename() == "PP-OCRv6_rec_medium.onnx");
    CHECK(fs::path(p.dict).filename() == "PP-OCRv6_rec_medium_dict.txt");
    CHECK(fs::path(p.det).parent_path() == fs::path("models"));
}

TEST_CASE("resolveModelPaths only rec override leaves others default") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.ocrVersion = "PP-OCRv6";
    cfg.modelType = "tiny";
    cfg.recModelPath = "custom/my_rec.onnx";
    ModelPaths p = resolveModelPaths(cfg);
    namespace fs = std::filesystem;
    CHECK(p.rec == "custom/my_rec.onnx");
    CHECK(fs::path(p.det).filename() == "PP-OCRv6_det.onnx");
    CHECK(fs::path(p.cls).filename() == "PP-OCRv6_cls.onnx");
    CHECK(fs::path(p.dict).filename() == "PP-OCRv6_rec_tiny_dict.txt");
}

TEST_CASE("resolveModelPaths all overrides win") {
    EngineConfig cfg;
    cfg.modelsDir = "models";
    cfg.detModelPath = "a/det.onnx";
    cfg.clsModelPath = "b/cls.onnx";
    cfg.recModelPath = "c/rec.onnx";
    cfg.dictPath = "d/dict.txt";
    ModelPaths p = resolveModelPaths(cfg);
    CHECK(p.det == "a/det.onnx");
    CHECK(p.cls == "b/cls.onnx");
    CHECK(p.rec == "c/rec.onnx");
    CHECK(p.dict == "d/dict.txt");
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

TEST_CASE("LinePrediction defaults include detScore") {
    LinePrediction lp;
    CHECK(lp.score == doctest::Approx(0.0f));
    CHECK(lp.detScore == doctest::Approx(0.0f));
}

TEST_CASE("toJson includes detScore") {
    LinePrediction line;
    line.text = "ab";
    line.score = 0.9f;
    line.detScore = 0.7f;
    line.polygon = {{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}, {7.f, 8.f}};
    const std::string json = toJson(line);
    CHECK(json.find("\"score\":0.9") != std::string::npos);
    CHECK(json.find("\"detScore\":0.7") != std::string::npos);
}

TEST_CASE("meanRecScore averages char scores and empty is 0") {
    CHECK(meanRecScore({}) == doctest::Approx(0.0f));
    CHECK(meanRecScore(std::vector<float>{0.5f, 1.0f}) == doctest::Approx(0.75f));
}

TEST_CASE("toJson serializes page with polygon, score, and escaped text") {
    PagePrediction page;
    page.image = "a\"b\\c";
    page.elapsedMs = 12.5f;
    page.lines.push_back(LinePrediction{
        {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}},
        "hello\nworld",
        0.91f,  // score (rec)
        0.0f,   // detScore
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
