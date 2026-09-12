// tests/test_engine.cpp
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

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
    CHECK(cfg.intraOpNumThreads == 0); // 0 = ORT decides
    CHECK(cfg.interOpNumThreads == 0); // 0 = ORT decides
    CHECK(cfg.useClahe == false);
    CHECK(cfg.splitOvermerged == false);
    CHECK(cfg.minDetBoxArea == doctest::Approx(20.0f)); // ppu's minimumAreaThreshold
    CHECK(cfg.minimumConfidence == doctest::Approx(0.5f));
    CHECK(cfg.spaceRecovery == false);
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

TEST_CASE("EngineConfig thread counts are settable and survive a copy") {
    EngineConfig cfg;
    cfg.intraOpNumThreads = 2;
    cfg.interOpNumThreads = 1;
    CHECK(cfg.intraOpNumThreads == 2);
    CHECK(cfg.interOpNumThreads == 1);

    // Callers routinely pass the config around by value (Engine stores its
    // own copy), so the knob has to travel with it.
    EngineConfig copy = cfg;
    CHECK(copy.intraOpNumThreads == 2);
    CHECK(copy.interOpNumThreads == 1);
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

namespace {

// Portable set/unset of ARBOOCR_OFFLINE. A null `value` means "unset":
// _putenv_s(name, "") *deletes* the variable on Windows (there is no
// unsetenv), so the empty string is the natural spelling of unset there and
// the helper never has to fake an empty-but-present variable.
void setOfflineEnv(const char* value) {
#ifdef _WIN32
    _putenv_s("ARBOOCR_OFFLINE", value ? value : "");
#else
    if (value) ::setenv("ARBOOCR_OFFLINE", value, 1);
    else ::unsetenv("ARBOOCR_OFFLINE");
#endif
}

} // namespace

TEST_CASE("modelDownloadsAllowed reads the flag and ARBOOCR_OFFLINE together") {
    const char* saved = std::getenv("ARBOOCR_OFFLINE");
    // Empty is folded into unset: the two are indistinguishable on Windows and
    // modelDownloadsAllowed() treats them the same anyway, so the restore at
    // the end round-trips on every platform.
    const bool hadVar = saved != nullptr && *saved != '\0';
    const std::string savedValue = hadVar ? std::string(saved) : std::string();

    EngineConfig cfg;
    REQUIRE(cfg.autoDownload); // the default this case is written against

    setOfflineEnv(nullptr);
    CHECK(modelDownloadsAllowed(cfg) == true);

    // --no-download / cfg.autoDownload = false forbids it on its own...
    cfg.autoDownload = false;
    CHECK(modelDownloadsAllowed(cfg) == false);
    // ...and wins outright: no environment value can put it back on the network.
    setOfflineEnv("0");
    CHECK(modelDownloadsAllowed(cfg) == false);
    setOfflineEnv("1");
    CHECK(modelDownloadsAllowed(cfg) == false);

    cfg.autoDownload = true;
    setOfflineEnv("1");
    CHECK(modelDownloadsAllowed(cfg) == false);

    // Set-but-"0" is an explicit "stay online", NOT offline. This pins the
    // exact semantics ensureOcrModels() has always had — anything looser would
    // silently take the network away from a wrapper that exports
    // ARBOOCR_OFFLINE=0 to disable offline mode.
    setOfflineEnv("0");
    CHECK(modelDownloadsAllowed(cfg) == true);

    // Any other non-empty value engages it, same as "1".
    setOfflineEnv("true");
    CHECK(modelDownloadsAllowed(cfg) == false);

    // doctest runs every case in one process and CI runs this binary with
    // ARBOOCR_OFFLINE=1, so leaking a value here would corrupt other cases.
    setOfflineEnv(hadVar ? savedValue.c_str() : nullptr);
    const char* restored = std::getenv("ARBOOCR_OFFLINE");
    CHECK(hadVar == (restored != nullptr && *restored != '\0'));
    if (hadVar) CHECK(savedValue == std::string(restored ? restored : ""));
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

TEST_CASE("toJson with backend includes backend field before image") {
    PagePrediction page;
    page.image = "page.jpg";
    page.elapsedMs = 12.5f;
    page.lines.push_back(LinePrediction{
        {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}},
        "hi",
        0.9f,
        0.8f,
    });

    const std::string json = toJson(page, std::string("cpu"));
    CHECK(json.find("\"backend\":\"cpu\"") != std::string::npos);
    CHECK(json.find("\"image\":\"page.jpg\"") != std::string::npos);
    // backend must come before image in the object
    CHECK(json.find("\"backend\"") < json.find("\"image\""));

    const std::string pretty = toJson(page, std::string("tensorrt"), true);
    CHECK(pretty.find("\"backend\":\"tensorrt\"") != std::string::npos);
    CHECK(pretty.find('\n') != std::string::npos);
}

TEST_CASE("toJson with backend on empty page is valid object") {
    PagePrediction empty;
    empty.image = "none.jpg";
    const std::string json = toJson(empty, std::string("cpu"));
    CHECK(json == "{\"backend\":\"cpu\",\"image\":\"none.jpg\",\"elapsedMs\":0,\"lines\":[]}");
}

// Engine::recognizeEncoded / recognizeEncodedAsync themselves need a
// constructed Engine (i.e. real ONNX files — Ort::Session throws otherwise), so
// they live in test_engine_inference.cpp territory. Everything new about them is
// the decode step; past that they hand off to the same runPipeline as
// recognize(const cv::Mat&). So test decodeImageBytes directly.

TEST_CASE("decodeImageBytes round-trips a PNG-encoded mat") {
    cv::Mat src(20, 30, CV_8UC3, cv::Scalar(10, 20, 30));
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", src, png));
    REQUIRE_FALSE(png.empty());

    cv::Mat decoded = decodeImageBytes(png.data(), png.size());
    REQUIRE_FALSE(decoded.empty());
    CHECK(decoded.rows == 20);
    CHECK(decoded.cols == 30);
    CHECK(decoded.channels() == 3);
    // PNG is lossless, so the pixels must survive the round trip exactly.
    CHECK(decoded.at<cv::Vec3b>(5, 5)[0] == 10);
    CHECK(decoded.at<cv::Vec3b>(5, 5)[1] == 20);
    CHECK(decoded.at<cv::Vec3b>(5, 5)[2] == 30);
}

TEST_CASE("decodeImageBytes always yields 3-channel BGR (IMREAD_COLOR)") {
    // A grayscale source must still come back as BGR — the pipeline downstream
    // (detector/getRotateCropImage) assumes 3 channels like cv::imread does.
    cv::Mat gray(8, 8, CV_8UC1, cv::Scalar(128));
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", gray, png));

    cv::Mat decoded = decodeImageBytes(png.data(), png.size());
    REQUIRE_FALSE(decoded.empty());
    CHECK(decoded.channels() == 3);
}

TEST_CASE("decodeImageBytes degrades to an empty mat instead of throwing") {
    // cv::imdecode CV_Asserts on an empty buffer, so these paths must never
    // reach it — an empty Mat is what recognizeEncoded turns into an
    // empty-lines PagePrediction, preserving recognize()'s never-throws contract.
    CHECK_NOTHROW(decodeImageBytes(nullptr, 0));
    CHECK(decodeImageBytes(nullptr, 0).empty());

    const std::vector<uint8_t> bytes(64, 0x41);
    CHECK_NOTHROW(decodeImageBytes(bytes.data(), 0));
    CHECK(decodeImageBytes(bytes.data(), 0).empty());

    CHECK_NOTHROW(decodeImageBytes(nullptr, 64));
    CHECK(decodeImageBytes(nullptr, 64).empty());

    // Non-image bytes: no codec magic number matches.
    const std::string garbage = "this is definitely not an image, not even close";
    const auto* junk = reinterpret_cast<const uint8_t*>(garbage.data());
    CHECK_NOTHROW(decodeImageBytes(junk, garbage.size()));
    CHECK(decodeImageBytes(junk, garbage.size()).empty());
}

TEST_CASE("decodeImageBytes does not throw on a truncated image") {
    // Valid PNG magic, payload cut off mid-stream: the codec is entered and may
    // fail deep inside. Only the never-throws contract is asserted — whether a
    // partial decode is salvaged is the codec's business, not ours.
    cv::Mat src(12, 12, CV_8UC3, cv::Scalar(255, 0, 0));
    std::vector<uchar> png;
    REQUIRE(cv::imencode(".png", src, png));
    REQUIRE(png.size() > 16);

    CHECK_NOTHROW(decodeImageBytes(png.data(), png.size() / 3));
    CHECK_NOTHROW(decodeImageBytes(png.data(), 8)); // magic number only
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
