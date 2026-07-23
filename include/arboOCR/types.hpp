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

// One recognized text line: polygon, decoded text, recognition confidence,
// and detector box score.
struct LinePrediction {
    Polygon polygon;
    std::string text;
    /// Mean CTC character confidence from the recognizer (0 if no chars).
    float score = 0.0f;
    /// Detector box score for this polygon (0 if unknown).
    float detScore = 0.0f;
};

/// Mean of per-character CTC scores; empty → 0. Used to fill LinePrediction::score.
inline float meanRecScore(const std::vector<float>& charScores) {
    if (charScores.empty()) return 0.0f;
    float sum = 0.0f;
    for (float s : charScores) sum += s;
    return sum / static_cast<float>(charScores.size());
}

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
