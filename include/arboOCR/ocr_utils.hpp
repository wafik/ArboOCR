#pragma once
// Adapted from RapidAI/RapidOcrOnnx's OcrUtils.h (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. Function bodies below are near-verbatim ports;
// only naming (unClip -> unClipBox) and the removal of unused
// debug/image-saving helpers differ from the original.

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "arboOCR/types.hpp"

namespace arbo::ocr {

/// Clamp x to [minVal, maxVal].
template <typename T>
inline T clampValue(T x, T minVal, T maxVal) {
    if (x > maxVal) return maxVal;
    if (x < minVal) return minVal;
    return x;
}

/// Compute the (srcWidth/Height, dstWidth/Height, ratios) for resizing `src`
/// so its longer side is at most `targetSize`, rounded down to a multiple
/// of 32 (DBNet's stride requirement). `targetSize` is a ceiling, not a
/// target: images already smaller are never upscaled (RapidOCR's
/// `Det.limit_type = max`). Dimensions still floor at 32 for DBNet's stride.
ScaleParam getScaleParam(const cv::Mat& src, int targetSize);

/// Order a RotatedRect's 4 corners into DBNet's expected
/// top-left/top-right/bottom-right/bottom-left-ish order. Also returns the
/// longer side length via `maxSideLen` (used as a minimum-size filter).
std::vector<cv::Point2f> getMinBoxes(const cv::RotatedRect& boxRect, float& maxSideLen);

/// Mean pixel value of `pred` inside the polygon `boxes` (used as the
/// DBNet box confidence score).
float boxScoreFast(const std::vector<cv::Point2f>& boxes, const cv::Mat& pred);

/// Expand a 4-point box outward by `unClipRatio` using Clipper's polygon
/// offset (DBNet's shrink-then-unclip box regression scheme).
cv::RotatedRect unClipBox(std::vector<cv::Point2f> box, float unClipRatio);

/// (pixel - mean) * norm per channel, laid out as CHW float32 for ONNX
/// tensor input.
std::vector<float> substractMeanNormalize(const cv::Mat& src, const float* meanVals, const float* normVals);

/// Perspective-crop + straighten one detected text box out of the full
/// image, ready for Classifier/Recognizer input. A crop that comes out tall
/// and narrow is rotated 90 degrees CCW so the recognizer reads *down* the
/// box rather than across it; optional `wasTransposed` reports whether that
/// happened (false on every early-return path). Callers that need to map
/// recognizer output back to page coordinates must pass it — the condition
/// lives here and only here, so it cannot drift.
cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box,
                           bool* wasTransposed = nullptr);

/// Rotate 180 degrees (used when Classifier detects upside-down text).
cv::Mat matRotateClockWise180(cv::Mat src);

/// Split wide det boxes that look like two side-by-side fields (common on
/// receipts: "TEL …" + "CO NO …" fused into one DBNet box). Uses a vertical
/// ink-gap in the perspective crop; returns the original box when no clean
/// valley is found. `minAspect` = crop width/height threshold (default 3.5).
/// `minGapDepth` = how much quieter the valley must be vs median column ink
/// (0–1; default 0.35). Pure geometry — no recognizer.
std::vector<RawTextBox> maybeSplitOvermergedBox(const RawTextBox& box,
                                                const cv::Mat& src,
                                                float minAspect = 3.5f,
                                                float minGapDepth = 0.35f);

/// Expand each box with maybeSplitOvermergedBox; order preserved (left then
/// right when split). Empty src → identity.
std::vector<RawTextBox> expandOvermergedBoxes(const std::vector<RawTextBox>& boxes,
                                             const cv::Mat& src);

/// Stable reading order: top-to-bottom then left-to-right by polygon centroid.
/// Lines are grouped into rows with a tolerance derived from the median
/// polygon height (half a line), so the ordering is the same whatever
/// resolution the page was scanned at.
void sortLinesReadingOrder(std::vector<LinePrediction>& lines);

/// CTC post: insert spaces where horizontal gaps between emitted tokens look
/// like whitespace (columnar receipts). `positions` are 0..1 fractions of the
/// crop width (timestep centers). Optional `scores` and `spans` stay
/// index-aligned; an injected space is given the gap it stands for as its
/// span, i.e. [previous token's end, next token's begin].
/// Mirrors ppu-paddle-ocr injectGapSpaces (median + 1.5/2.5 quanta).
void injectGapSpaces(std::vector<std::string>& tokens,
                     std::vector<float>& positions,
                     std::vector<float>* scores = nullptr,
                     std::vector<TokenSpan>* spans = nullptr);

/// Map a horizontal span of a recognizer crop back to a polygon in
/// source-image coordinates. `lineQuad` is a LinePrediction::polygon: exactly
/// 4 points, 0=TL 1=TR 2=BR 3=BL (getMinBoxes' order). `wasTransposed` /
/// `wasRotated180` are the two orientation changes the crop went through
/// (getRotateCropImage's 90-degree CCW rotation and the classifier's 180 flip)
/// and must be reported by those steps, not re-derived here.
/// Returns the span's quad in the same TL,TR,BR,BL page order, or an empty
/// polygon if `lineQuad` does not have 4 points. Spans are clamped to [0,1]
/// and a reversed span (begin > end) is swapped.
Polygon spanToPolygon(const Polygon& lineQuad, TokenSpan span,
                      bool wasTransposed, bool wasRotated180);

/// Group decoded tokens into word boxes. Tokens are split on space tokens;
/// CJK tokens each become their own word (no spaces to split on). The three
/// token vectors must be index-aligned — mismatched sizes return empty rather
/// than reading out of bounds. Each word's span runs from its first token's
/// begin to its last token's end, its score is the mean of its tokens' scores,
/// and its polygon comes from spanToPolygon.
std::vector<WordBox> groupTokensIntoWords(const std::vector<std::string>& tokens,
                                          const std::vector<TokenSpan>& spans,
                                          const std::vector<float>& scores,
                                          const Polygon& lineQuad,
                                          bool wasTransposed,
                                          bool wasRotated180);

/// Collapse space runs; map fullwidth ASCII/ideographic space → halfwidth when
/// the string has no CJK. In-place. Mirrors ppu refineDecodedChars.
void refineDecodedText(std::string& text);

/// drop_score gate: alphanumeric text needs `minimumConfidence`; pure
/// symbol/punct needs minimumConfidence+0.3 (capped at 1). `minimumConfidence
/// <= 0` always keeps. Mirrors ppu BaseRecognitionService filter.
bool keepByConfidence(const std::string& text, float confidence, float minimumConfidence);

} // namespace arbo::ocr
