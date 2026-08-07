// Original arboOCR code (not a RapidOcrOnnx port) — see visualize.hpp.
#include "arboOCR/visualize.hpp"

#include <algorithm>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace arbo::ocr {

namespace {

// Same bar EngineConfig::minimumConfidence defaults to (engine.hpp). Not
// pulled in as a shared constant on purpose: a leaf drawing file shouldn't
// include the ORT-heavy engine.hpp for one float.
constexpr float kConfidenceBar = 0.5f;

// Detector polygons are floats; a wild value would make the cast to int UB and
// OpenCV's fixed-point drawing unhappy. 1e6 px is ~1000x any real page and far
// inside int32, so clamping here costs nothing real and leaves the actual
// off-image clipping to cv::polylines, which does it per segment.
constexpr float kCoordLimit = 1.0e6f;

constexpr int kThickness = 2;

cv::Point toCvPoint(const Point2f& p) {
    return cv::Point(static_cast<int>(std::clamp(p.x, -kCoordLimit, kCoordLimit)),
                     static_cast<int>(std::clamp(p.y, -kCoordLimit, kCoordLimit)));
}

} // namespace

cv::Mat drawResult(const cv::Mat& image, const PagePrediction& page) {
    if (image.empty()) return cv::Mat();

    cv::Mat out;
    if (image.channels() == 1) {
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
    } else {
        out = image.clone();
    }

    const cv::Scalar confident(0, 255, 0); // BGR green
    const cv::Scalar doubtful(0, 0, 255);  // BGR red

    std::vector<cv::Point> pts;
    for (const auto& line : page.lines) {
        // ponytail: a 0- or 1-point "polygon" has no edge to outline; skipping
        // beats trusting OpenCV's degenerate-contour handling.
        if (line.polygon.size() < 2) continue;
        pts.clear();
        pts.reserve(line.polygon.size());
        for (const auto& p : line.polygon) pts.push_back(toCvPoint(p));
        cv::polylines(out, pts, /*isClosed=*/true,
                      line.score < kConfidenceBar ? doubtful : confident, kThickness);
    }
    return out;
}

} // namespace arbo::ocr
