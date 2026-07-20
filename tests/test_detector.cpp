// tests/test_detector.cpp
#include <doctest/doctest.h>
#include "arboOCR/detector.hpp"

using namespace arbo::ocr;

TEST_CASE("Detector default-constructs without a loaded model") {
    Detector net;
    cv::Mat blank = cv::Mat::zeros(100, 100, CV_8UC3);
    ScaleParam scale{100, 100, 96, 96, 0.96f, 0.96f};
    auto boxes = net.getTextBoxes(blank, scale, 0.5f, 0.3f, 1.6f);
    CHECK(boxes.empty());
}
