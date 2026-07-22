#pragma once
// Adapted from RapidAI/RapidOcrOnnx (Apache-2.0) — see THIRD_PARTY_NOTICES.md.
// Struct shapes match the original; renamed to arboOCR conventions.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace arbo::ocr {

struct ScaleParam {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    float ratioWidth;
    float ratioHeight;
};

// One detected text box before recognition (Detector output).
struct RawTextBox {
    std::vector<cv::Point> boxPoint; // 4 points, clockwise from top-left-ish
    float score;
};

// Classifier orientation result for one cropped text-line image.
struct RawAngle {
    int index;  // 0 or 1 (0deg / 180deg); -1 if angle classification was skipped
    float score;
};

// Recognizer result for one cropped text-line image.
struct RawTextLine {
    std::string text;
    std::vector<float> charScores;
};

// A single polygon point, used to report detected line geometry to callers
// without pulling in OpenCV types at the public API boundary.
struct Point2f {
    float x;
    float y;
};
using Polygon = std::vector<Point2f>;

// One recognized text line: its polygon, decoded text, and detection score.
struct LinePrediction {
    Polygon polygon;
    std::string text;
    float score = 0.0f;
};

// Full-page recognition result.
struct PagePrediction {
    std::string image;
    std::vector<LinePrediction> lines;
    float elapsedMs = 0.0f;
};

/// Serialize a page prediction (text, scores, polygon points, timing) to JSON.
/// Never throws. When `pretty` is true, uses multi-line indented output.
std::string toJson(const PagePrediction& page, bool pretty = false);

/// Serialize a single line prediction to a JSON object string.
std::string toJson(const LinePrediction& line, bool pretty = false);

} // namespace arbo::ocr
