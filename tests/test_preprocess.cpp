#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "arboOCR/preprocess.hpp"

using namespace arbo::ocr;

namespace {
double stddev(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    cv::Scalar mean, stddevScalar;
    cv::meanStdDev(gray, mean, stddevScalar);
    return stddevScalar[0];
}
} // namespace

TEST_CASE("applyClahe increases contrast on a low-contrast 3-channel image") {
    // Two mid-gray blocks with a small step between them -> low stddev.
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(120, 120, 120));
    img(cv::Rect(0, 0, 100, 50)).setTo(cv::Scalar(130, 130, 130));
    double before = stddev(img);

    cv::Mat out = applyClahe(img);
    REQUIRE_FALSE(out.empty());
    CHECK(out.channels() == 3);
    CHECK(out.rows == img.rows);
    CHECK(out.cols == img.cols);
    double after = stddev(out);
    CHECK(after > before);
}

TEST_CASE("applyClahe on an empty Mat returns empty, does not throw") {
    cv::Mat empty;
    cv::Mat out;
    CHECK_NOTHROW(out = applyClahe(empty));
    CHECK(out.empty());
}

TEST_CASE("applyClahe handles a 1-channel (grayscale) image without throwing") {
    cv::Mat gray(50, 50, CV_8UC1, cv::Scalar(100));
    gray(cv::Rect(0, 0, 50, 25)).setTo(cv::Scalar(110));
    cv::Mat out;
    CHECK_NOTHROW(out = applyClahe(gray));
    REQUIRE_FALSE(out.empty());
    CHECK(out.channels() == 1);
    CHECK(out.rows == 50);
    CHECK(out.cols == 50);
}

TEST_CASE("applyClahe on an unsupported channel count returns input unchanged") {
    cv::Mat img4(20, 20, CV_8UC4, cv::Scalar(10, 20, 30, 40));
    cv::Mat out = applyClahe(img4);
    CHECK(out.channels() == 4);
    CHECK(out.rows == 20);
    CHECK(out.cols == 20);
}

TEST_CASE("applyClahe on an unsupported depth (float) returns input unchanged, does not throw") {
    cv::Mat img(20, 20, CV_32FC3, cv::Scalar(0.1f, 0.2f, 0.3f));
    cv::Mat out;
    CHECK_NOTHROW(out = applyClahe(img));
    CHECK(out.depth() == CV_32F);
    CHECK(out.channels() == 3);
    CHECK(out.rows == 20);
    CHECK(out.cols == 20);
}
