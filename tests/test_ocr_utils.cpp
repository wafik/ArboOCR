#include <algorithm>

#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "arboOCR/ocr_utils.hpp"

using namespace arbo::ocr;

TEST_CASE("getScaleParam rounds dims down to a multiple of 32") {
    cv::Mat img(400, 600, CV_8UC3); // 600 wide x 400 tall
    ScaleParam s = getScaleParam(img, 736);
    CHECK(s.srcWidth == 600);
    CHECK(s.srcHeight == 400);
    // 600 is already under the 736 ceiling, so both dims only floor to /32.
    CHECK(s.dstWidth % 32 == 0);
    CHECK(s.dstHeight % 32 == 0);
    CHECK(s.dstWidth > 0);
    CHECK(s.dstHeight > 0);
}

TEST_CASE("getScaleParam never upscales a small image (targetSize is a ceiling)") {
    cv::Mat thumb(150, 200, CV_8UC3); // 200 wide x 150 tall, far under the limit
    ScaleParam s = getScaleParam(thumb, 960);
    CHECK(s.srcWidth == 200);
    CHECK(s.srcHeight == 150);
    // Old behaviour scaled the long side up to 960 and paid detector cost on
    // interpolated pixels; the ceiling semantic only floors to /32.
    CHECK(s.dstWidth == 192);
    CHECK(s.dstHeight == 128);
    CHECK(s.dstWidth <= s.srcWidth);
    CHECK(s.dstHeight <= s.srcHeight);
    CHECK(s.ratioWidth <= 1.0f);
    CHECK(s.ratioHeight <= 1.0f);
}

TEST_CASE("getScaleParam still downscales a large image to the ceiling") {
    cv::Mat page(1080, 1920, CV_8UC3); // 1920 wide x 1080 tall
    ScaleParam s = getScaleParam(page, 960);
    CHECK(s.srcWidth == 1920);
    CHECK(s.srcHeight == 1080);
    CHECK(s.dstWidth == 960);  // long side exactly at the ceiling
    CHECK(s.dstHeight == 512); // 540 floored to a multiple of 32
    CHECK(s.dstWidth % 32 == 0);
    CHECK(s.dstHeight % 32 == 0);
    CHECK(std::max(s.dstWidth, s.dstHeight) <= 960);
}

TEST_CASE("getMinBoxes returns 4 ordered points and correct maxSideLen") {
    cv::RotatedRect rect(cv::Point2f(50, 50), cv::Size2f(100, 40), 0.0f);
    float maxSideLen = 0.0f;
    auto box = getMinBoxes(rect, maxSideLen);
    REQUIRE(box.size() == 4);
    CHECK(maxSideLen == doctest::Approx(100.0f));
}

TEST_CASE("boxScoreFast returns higher score for a bright region under the box") {
    cv::Mat pred = cv::Mat::zeros(100, 100, CV_32F);
    pred(cv::Rect(10, 10, 30, 30)).setTo(0.9f); // bright square
    std::vector<cv::Point2f> box = {
        {10, 10}, {40, 10}, {40, 40}, {10, 40}
    };
    float score = boxScoreFast(box, pred);
    CHECK(score > 0.8f);
}

TEST_CASE("unClipBox expands a box outward (result area >= input area)") {
    std::vector<cv::Point2f> box = {
        {0, 0}, {100, 0}, {100, 40}, {0, 40}
    };
    cv::RotatedRect result = unClipBox(box, 1.6f);
    CHECK(result.size.width * result.size.height >= 100.0f * 40.0f);
}

TEST_CASE("substractMeanNormalize produces correct output length") {
    cv::Mat img = cv::Mat::ones(10, 20, CV_8UC3) * 128; // 20 wide, 10 tall, 3 channels
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float norm[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    auto values = substractMeanNormalize(img, mean, norm);
    CHECK(values.size() == 10 * 20 * 3);
}

TEST_CASE("matRotateClockWise180 flips both axes") {
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    img.at<uchar>(0, 0) = 255; // top-left corner marked
    cv::Mat rotated = matRotateClockWise180(img.clone());
    CHECK(rotated.at<uchar>(9, 9) == 255); // should land at bottom-right
}

TEST_CASE("matRotateClockWise180 on an empty Mat returns empty, does not throw") {
    cv::Mat empty;
    CHECK_NOTHROW(matRotateClockWise180(empty).empty());
}

TEST_CASE("getRotateCropImage returns empty Mat for a degenerate (zero-area) box, does not throw") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    // All 4 points identical -> zero width/height after the crop math.
    std::vector<cv::Point> degenerateBox = {{10, 10}, {10, 10}, {10, 10}, {10, 10}};
    cv::Mat result;
    CHECK_NOTHROW(result = getRotateCropImage(src, degenerateBox));
    CHECK(result.empty());
}

TEST_CASE("getRotateCropImage returns empty Mat for a box outside image bounds, does not throw") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    std::vector<cv::Point> outOfBoundsBox = {{200, 200}, {250, 200}, {250, 250}, {200, 250}};
    cv::Mat result;
    CHECK_NOTHROW(result = getRotateCropImage(src, outOfBoundsBox));
    CHECK(result.empty());
}

TEST_CASE("maybeSplitOvermergedBox splits wide crop with ink gap") {
    // White background, two dark text blocks with a clear gap in the middle.
    cv::Mat src(40, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(src, cv::Rect(5, 5, 70, 30), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(src, cv::Rect(125, 5, 70, 30), cv::Scalar(20, 20, 20), cv::FILLED);
    RawTextBox box;
    box.boxPoint = {{0, 0}, {200, 0}, {200, 40}, {0, 40}};
    box.score = 0.9f;
    auto parts = maybeSplitOvermergedBox(box, src, /*minAspect=*/3.0f, /*minGapDepth=*/0.3f);
    REQUIRE(parts.size() == 2);
    CHECK(parts[0].score == doctest::Approx(0.9f));
    CHECK(parts[1].score == doctest::Approx(0.9f));
}

TEST_CASE("maybeSplitOvermergedBox keeps short box intact") {
    cv::Mat src(40, 60, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(src, cv::Rect(5, 5, 50, 30), cv::Scalar(20, 20, 20), cv::FILLED);
    RawTextBox box;
    box.boxPoint = {{0, 0}, {60, 0}, {60, 40}, {0, 40}};
    box.score = 0.8f;
    auto parts = maybeSplitOvermergedBox(box, src);
    REQUIRE(parts.size() == 1);
}

TEST_CASE("sortLinesReadingOrder sorts by y then x") {
    LinePrediction a, b, c;
    a.polygon = {{100, 10}, {140, 10}, {140, 20}, {100, 20}};
    a.text = "right-top";
    b.polygon = {{10, 10}, {50, 10}, {50, 20}, {10, 20}};
    b.text = "left-top";
    c.polygon = {{10, 40}, {50, 40}, {50, 50}, {10, 50}};
    c.text = "bottom";
    std::vector<LinePrediction> lines = {a, c, b};
    sortLinesReadingOrder(lines);
    REQUIRE(lines.size() == 3);
    CHECK(lines[0].text == "left-top");
    CHECK(lines[1].text == "right-top");
    CHECK(lines[2].text == "bottom");
}

TEST_CASE("sortLinesReadingOrder gives the same order at 1x and 4x coordinates") {
    // Same logical page at two resolutions. The top row's two columns sit 4px
    // apart in y at 1x — 16px at 4x, which is past the old hard-coded 12px
    // tolerance, so the high-DPI copy used to fragment that row and emit the
    // right-hand column first. With a tolerance derived from median line
    // height the ordering is identical at both scales.
    auto buildPage = [](float k) {
        auto rect = [k](float x0, float y0, float x1, float y1, const char* text) {
            LinePrediction line;
            line.polygon = {{x0 * k, y0 * k}, {x1 * k, y0 * k}, {x1 * k, y1 * k}, {x0 * k, y1 * k}};
            line.text = text;
            return line;
        };
        // Deliberately shuffled input order.
        return std::vector<LinePrediction>{
            rect(100, 6, 140, 20, "row0-right"),
            rect(10, 70, 50, 84, "row2-left"),
            rect(10, 10, 50, 24, "row0-left"),
            rect(100, 41, 140, 55, "row1-right"),
            rect(10, 40, 50, 54, "row1-left"),
        };
    };
    const std::vector<std::string> expected = {
        "row0-left", "row0-right", "row1-left", "row1-right", "row2-left"};

    for (float k : {1.0f, 4.0f}) {
        CAPTURE(k);
        std::vector<LinePrediction> lines = buildPage(k);
        sortLinesReadingOrder(lines);
        REQUIRE(lines.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK(lines[i].text == expected[i]);
        }
    }
}

TEST_CASE("sortLinesReadingOrder handles empty, single and zero-height input") {
    std::vector<LinePrediction> none;
    CHECK_NOTHROW(sortLinesReadingOrder(none));
    CHECK(none.empty());

    LinePrediction only;
    only.polygon = {{10, 10}, {50, 10}, {50, 24}, {10, 24}};
    only.text = "solo";
    std::vector<LinePrediction> one = {only};
    CHECK_NOTHROW(sortLinesReadingOrder(one));
    REQUIRE(one.size() == 1);
    CHECK(one[0].text == "solo");

    // Degenerate: zero-height polygons (median height 0 -> tolerance floor)
    // plus an empty polygon (centroid 0,0). Must still order top-to-bottom.
    LinePrediction flatLow, flatHigh, noPoly;
    flatLow.polygon = {{10, 60}, {50, 60}, {50, 60}, {10, 60}};
    flatLow.text = "low";
    flatHigh.polygon = {{10, 20}, {50, 20}, {50, 20}, {10, 20}};
    flatHigh.text = "high";
    noPoly.text = "no-polygon";
    std::vector<LinePrediction> degenerate = {flatLow, noPoly, flatHigh};
    CHECK_NOTHROW(sortLinesReadingOrder(degenerate));
    REQUIRE(degenerate.size() == 3);
    CHECK(degenerate[0].text == "no-polygon");
    CHECK(degenerate[1].text == "high");
    CHECK(degenerate[2].text == "low");
}

TEST_CASE("injectGapSpaces inserts space on wide cross-class gap") {
    // positions with a clear column gap between letter and digit clusters
    std::vector<std::string> tokens = {"A", "B", "C", "D", "1", "2", "3", "4"};
    std::vector<float> pos = {0.05f, 0.10f, 0.15f, 0.20f, 0.70f, 0.75f, 0.80f, 0.85f};
    std::vector<float> scores(8, 0.9f);
    injectGapSpaces(tokens, pos, &scores);
    bool foundSpace = false;
    for (const auto& t : tokens) if (t == " ") foundSpace = true;
    CHECK(foundSpace);
    CHECK(tokens.size() == scores.size());
    CHECK(tokens.size() == pos.size());
}

TEST_CASE("refineDecodedText maps fullwidth colon and collapses spaces") {
    // U+FF1A fullwidth colon = UTF-8 EF BC 9A (avoid \x in string literal for MSVC)
    std::string s = "Tel";
    s += static_cast<char>(0xEF);
    s += static_cast<char>(0xBC);
    s += static_cast<char>(0x9A);
    s += "07  88";
    refineDecodedText(s);
    CHECK(s == "Tel:07 88");
}

TEST_CASE("keepByConfidence matches ppu dual bar") {
    CHECK(keepByConfidence("TOTAL", 0.6f, 0.5f));
    CHECK_FALSE(keepByConfidence("TOTAL", 0.4f, 0.5f));
    CHECK(keepByConfidence("+-", 0.85f, 0.5f)); // symbol bar 0.8
    CHECK_FALSE(keepByConfidence("+-", 0.7f, 0.5f));
    CHECK(keepByConfidence("x", 0.1f, 0.0f)); // disabled
}

TEST_CASE("getRotateCropImage returns empty Mat for an empty source image, does not throw") {
    cv::Mat empty;
    std::vector<cv::Point> box = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    cv::Mat result;
    CHECK_NOTHROW(result = getRotateCropImage(empty, box));
    CHECK(result.empty());
}

TEST_CASE("getRotateCropImage still crops a valid box correctly (regression guard)") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    src(cv::Rect(10, 10, 30, 30)).setTo(cv::Scalar(255, 255, 255));
    std::vector<cv::Point> box = {{10, 10}, {40, 10}, {40, 40}, {10, 40}};
    cv::Mat result = getRotateCropImage(src, box);
    CHECK_FALSE(result.empty());
    CHECK(result.cols > 0);
    CHECK(result.rows > 0);
}
