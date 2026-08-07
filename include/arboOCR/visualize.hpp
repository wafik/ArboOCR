#pragma once
// Original arboOCR code (not a RapidOcrOnnx port) — the debug-drawing helper
// the README hand-waves as "draw boxes with e.g. cv::polylines".

#include <opencv2/core.hpp>

#include "arboOCR/types.hpp"

namespace arbo::ocr {

/// Return a copy of `image` with every line's polygon outlined. The caller's
/// Mat is never modified and the result never shares its buffer.
///
/// Lines at or above 0.5 recognition confidence are outlined green, lines
/// below it red. 0.5 is not a new number: it is EngineConfig::minimumConfidence's
/// default (engine.hpp), the bar the pipeline itself uses to drop lines — so a
/// red box is one the default config would have discarded, and you only ever
/// see one after lowering or disabling that filter, which is exactly when you
/// are debugging.
///
/// A 1-channel input is promoted to BGR so those two colors stay
/// distinguishable (on a gray Mat both would collapse to channel 0); any other
/// input yields a Mat of the same size and type.
///
/// Never throws. An empty image returns an empty Mat, empty `page.lines`
/// returns a plain copy, polygons with fewer than 2 points are skipped, and
/// off-image coordinates are clipped by OpenCV's own drawing code.
///
/// ponytail: boxes only — cv::putText can't render CJK (Hershey fonts are a
/// Latin subset) and arboOCR's PP-OCRv6 models are multilingual by default.
/// Text overlay needs opencv_freetype plus a bundled TTF; RapidOCR solves it by
/// downloading a font at runtime, which is not a dependency this library takes.
cv::Mat drawResult(const cv::Mat& image, const PagePrediction& page);

} // namespace arbo::ocr
