#pragma once
// Original arboOCR code (not a RapidOcrOnnx port) — see ocr_utils.hpp for
// the ported preprocessing helpers this file is deliberately separate from.

#include <opencv2/core.hpp>

namespace arbo::ocr {

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to boost
/// local contrast, e.g. for faded/low-contrast scanned documents. For a
/// 3-channel image, operates on the L channel of Lab color space (preserves
/// hue/saturation) and converts back to BGR. For a 1-channel image, applies
/// CLAHE directly. Fixed clipLimit=2.0, tileGridSize=8x8 (OpenCV's
/// tileGridSize default; clipLimit chosen for OCR use, not OpenCV's own
/// default of 40.0). Never throws: empty Mat, an unsupported channel count
/// (not 1 or 3), or an unsupported depth (not CV_8U) is returned unchanged.
cv::Mat applyClahe(const cv::Mat& src);

} // namespace arbo::ocr
