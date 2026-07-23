#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "arboOCR/ocr_utils.hpp"

using namespace arbo::ocr;

TEST_CASE("getScaleParam scales to target size, rounds to multiple of 32") {
    cv::Mat img(400, 600, CV_8UC3); // 600 wide x 400 tall
    ScaleParam s = getScaleParam(img, 736);
    CHECK(s.srcWidth == 600);
    CHECK(s.srcHeight == 400);
    // Wider side (600) scales toward 736; dstWidth should be a multiple of 32
    CHECK(s.dstWidth % 32 == 0);
    CHECK(s.dstHeight % 32 == 0);
    CHECK(s.dstWidth > 0);
    CHECK(s.dstHeight > 0);
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
