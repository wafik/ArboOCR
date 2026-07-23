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
/// so its longer side approaches `targetSize`, rounded down to a multiple
/// of 32 (DBNet's stride requirement).
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
/// image, ready for Classifier/Recognizer input.
cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box);

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
void sortLinesReadingOrder(std::vector<LinePrediction>& lines);

} // namespace arbo::ocr
