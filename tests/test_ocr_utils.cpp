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
