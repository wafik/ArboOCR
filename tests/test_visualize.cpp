#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "arboOCR/visualize.hpp"

using namespace arbo::ocr;

namespace {

// Axis-aligned rectangle as a 4-point polygon, clockwise from top-left.
LinePrediction makeRect(float x0, float y0, float x1, float y1, float score) {
    LinePrediction line;
    line.polygon = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    line.text = "hello";
    line.score = score;
    line.detScore = 0.9f;
    return line;
}

// Count of differing channel values between two same-size/type Mats.
int diffCount(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(diff.reshape(1));
}

// Exact `color` present within `radius` px of (x, y)? Tolerant of how OpenCV
// centers a 2px-thick stroke, but still asserts the exact drawn color.
bool hasColorNear(const cv::Mat& img, int x, int y, const cv::Vec3b& color, int radius = 2) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const int yy = y + dy;
            const int xx = x + dx;
            if (yy < 0 || xx < 0 || yy >= img.rows || xx >= img.cols) continue;
            if (img.at<cv::Vec3b>(yy, xx) == color) return true;
        }
    }
    return false;
}

const cv::Vec3b kGreen(0, 255, 0);
const cv::Vec3b kRed(0, 0, 255);
const cv::Vec3b kWhite(255, 255, 255);

} // namespace

TEST_CASE("drawResult outlines a known polygon onto the returned image") {
    cv::Mat white(80, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    PagePrediction page;
    page.lines.push_back(makeRect(10.f, 10.f, 60.f, 40.f, 0.9f));

    cv::Mat out = drawResult(white, page);
    REQUIRE_FALSE(out.empty());
    CHECK(out.rows == white.rows);
    CHECK(out.cols == white.cols);
    CHECK(out.type() == white.type());

    // Something was drawn at all...
    CHECK(diffCount(white, out) > 0);
    // ...and specifically on the rectangle's top edge, at x=35, y=10.
    const cv::Vec3b edge = out.at<cv::Vec3b>(10, 35);
    const bool edgeChanged = (edge != kWhite);
    CHECK(edgeChanged);
    // Interior stays untouched — this is an outline, not a fill.
    const bool interiorUntouched = (out.at<cv::Vec3b>(25, 35) == kWhite);
    CHECK(interiorUntouched);
}

TEST_CASE("drawResult does not modify the caller's image") {
    cv::Mat white(80, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat before = white.clone();
    PagePrediction page;
    page.lines.push_back(makeRect(5.f, 5.f, 70.f, 50.f, 0.9f));

    cv::Mat out = drawResult(white, page);
    CHECK(diffCount(before, white) == 0); // input untouched
    CHECK(diffCount(before, out) > 0);    // output actually drawn on
    CHECK(out.data != white.data);        // and not an alias of the input
}

TEST_CASE("drawResult with no lines returns an unchanged copy") {
    cv::Mat img(30, 40, CV_8UC3, cv::Scalar(10, 20, 30));
    PagePrediction page; // page.lines is empty

    cv::Mat out = drawResult(img, page);
    REQUIRE_FALSE(out.empty());
    CHECK(out.rows == img.rows);
    CHECK(out.cols == img.cols);
    CHECK(out.type() == img.type());
    CHECK(diffCount(img, out) == 0);
    CHECK(out.data != img.data);
}

TEST_CASE("drawResult colors sub-threshold lines differently from confident ones") {
    cv::Mat white(120, 120, CV_8UC3, cv::Scalar(255, 255, 255));
    PagePrediction page;
    page.lines.push_back(makeRect(10.f, 10.f, 50.f, 40.f, 0.95f)); // above the 0.5 bar
    page.lines.push_back(makeRect(10.f, 70.f, 50.f, 100.f, 0.20f)); // below the 0.5 bar

    cv::Mat out = drawResult(white, page);
    REQUIRE_FALSE(out.empty());
    CHECK(hasColorNear(out, 30, 10, kGreen));
    CHECK(hasColorNear(out, 30, 70, kRed));
}

TEST_CASE("drawResult promotes a 1-channel image to BGR so both colors survive") {
    cv::Mat gray(40, 60, CV_8UC1, cv::Scalar(255));
    PagePrediction page;
    page.lines.push_back(makeRect(5.f, 5.f, 50.f, 30.f, 0.9f));

    cv::Mat out = drawResult(gray, page);
    REQUIRE_FALSE(out.empty());
    CHECK(out.channels() == 3);
    CHECK(out.rows == 40);
    CHECK(out.cols == 60);
    CHECK(hasColorNear(out, 25, 5, kGreen));
}

TEST_CASE("drawResult on an empty Mat returns empty, does not throw") {
    cv::Mat empty;
    PagePrediction page;
    page.lines.push_back(makeRect(1.f, 1.f, 10.f, 10.f, 0.9f));

    cv::Mat out;
    CHECK_NOTHROW(out = drawResult(empty, page));
    CHECK(out.empty());
}

TEST_CASE("drawResult survives degenerate polygons and out-of-bounds coordinates") {
    PagePrediction page;

    page.lines.push_back(LinePrediction{}); // 0-point polygon

    LinePrediction onePoint;
    onePoint.polygon = {{5.f, 5.f}};
    onePoint.score = 0.9f;
    page.lines.push_back(onePoint);

    LinePrediction twoPoint; // degenerate "polygon" — a bare segment
    twoPoint.polygon = {{1.f, 1.f}, {20.f, 20.f}};
    twoPoint.score = 0.9f;
    page.lines.push_back(twoPoint);

    // Straddling the image, entirely past it, and entirely before it.
    page.lines.push_back(makeRect(-9000.f, -9000.f, 90000.f, 90000.f, 0.9f));
    page.lines.push_back(makeRect(5000.f, 5000.f, 6000.f, 6000.f, 0.9f));
    page.lines.push_back(makeRect(-6000.f, -6000.f, -5000.f, -5000.f, 0.1f));

    cv::Mat img(50, 50, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat out;
    CHECK_NOTHROW(out = drawResult(img, page));
    REQUIRE_FALSE(out.empty());
    CHECK(out.rows == 50);
    CHECK(out.cols == 50);
    CHECK(diffCount(img, out) > 0); // the 2-point segment is still on-image

    cv::Mat emptyOut;
    CHECK_NOTHROW(emptyOut = drawResult(cv::Mat(), page));
    CHECK(emptyOut.empty());
}
