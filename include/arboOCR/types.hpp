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

/// Horizontal extent of one CTC token, as fractions of the recognizer crop's
/// *content* width — i.e. batch padding is already divided out. begin/end are
/// in [0,1] with begin <= end. See Recognizer::scoreToTextLine.
struct TokenSpan {
    float begin = 0.0f;
    float end = 0.0f;
};

// Recognizer result for one cropped text-line image.
struct RawTextLine {
    std::string text;
    std::vector<float> charScores;
    /// One entry per emitted token, index-aligned with charScores. Populated
    /// only when word boxes were requested — computing spans costs nothing,
    /// but carrying them for every line of every page is not free.
    std::vector<std::string> tokens;
    std::vector<TokenSpan> spans;
};

// A single polygon point, used to report detected line geometry to callers
// without pulling in OpenCV types at the public API boundary.
struct Point2f {
    float x;
    float y;
};
using Polygon = std::vector<Point2f>;

/// One word (or one CJK character) inside a recognized line, with its own
/// polygon in source-image coordinates. Scripts that delimit words with
/// spaces group into words; CJK characters stand alone, because there is no
/// space to split on — this mirrors RapidOCR's return_word_box behaviour.
///
/// Accuracy: polygons are derived from CTC timestep alignment, which is a
/// by-product of recognition rather than a supervised output. CTC peaks
/// partway through a glyph, so boxes run roughly half a character wide of
/// true glyph extents — measured against synthetic text, consistently
/// lagging right. Good enough for highlighting, search hit-marking, and
/// reading-order reconstruction; not good enough to crop a glyph from.
/// Widening each span to meet its neighbours would recover the extents, but
/// picking that constant needs labelled data we do not have.
struct WordBox {
    Polygon polygon;
    std::string text;
    /// Mean CTC confidence over this word's characters.
    float score = 0.0f;
};

// One recognized text line: polygon, decoded text, recognition confidence,
// and detector box score.
struct LinePrediction {
    Polygon polygon;
    std::string text;
    /// Mean CTC character confidence from the recognizer (0 if no chars).
    float score = 0.0f;
    /// Detector box score for this polygon (0 if unknown).
    float detScore = 0.0f;
    /// Per-word polygons. Empty unless EngineConfig::returnWordBoxes is set.
    std::vector<WordBox> words;
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

/// Same as toJson(page, pretty) but includes a "backend" field (e.g.
/// "cpu"/"cuda"/"tensorrt") — Engine::backend() isn't part of PagePrediction,
/// so callers that want it in the JSON (e.g. the CLI's --json mode) use this
/// overload instead of splicing it in themselves.
std::string toJson(const PagePrediction& page, const std::string& backend, bool pretty = false);

/// Serialize a single line prediction to a JSON object string.
std::string toJson(const LinePrediction& line, bool pretty = false);

} // namespace arbo::ocr
