// Adapted from RapidAI/RapidOcrOnnx's OcrUtils.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. getScaleParam/getMinBoxes/boxScoreFast/
// unClipBox/substractMeanNormalize/getRotateCropImage/matRotateClockWise180
// are near-verbatim ports of the corresponding upstream functions.
#include "arboOCR/ocr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "clipper.hpp" // vendored, see vendor/rapidocronnx/

namespace arbo::ocr {

ScaleParam getScaleParam(const cv::Mat& src, int targetSize) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;
    int dstWidth = srcWidth;
    int dstHeight = srcHeight;

    float ratio = (srcWidth > srcHeight)
        ? static_cast<float>(targetSize) / static_cast<float>(srcWidth)
        : static_cast<float>(targetSize) / static_cast<float>(srcHeight);

    dstWidth = static_cast<int>(static_cast<float>(srcWidth) * ratio);
    dstHeight = static_cast<int>(static_cast<float>(srcHeight) * ratio);
    if (dstWidth % 32 != 0) {
        dstWidth = std::max((dstWidth / 32) * 32, 32);
    }
    if (dstHeight % 32 != 0) {
        dstHeight = std::max((dstHeight / 32) * 32, 32);
    }
    float ratioWidth = static_cast<float>(dstWidth) / static_cast<float>(srcWidth);
    float ratioHeight = static_cast<float>(dstHeight) / static_cast<float>(srcHeight);
    return {srcWidth, srcHeight, dstWidth, dstHeight, ratioWidth, ratioHeight};
}

namespace {
bool pointCompareX(const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; }
}

std::vector<cv::Point2f> getMinBoxes(const cv::RotatedRect& boxRect, float& maxSideLen) {
    maxSideLen = std::max(boxRect.size.width, boxRect.size.height);
    cv::Point2f vertices[4];
    boxRect.points(vertices);
    std::vector<cv::Point2f> boxPoint(vertices, vertices + 4);
    std::sort(boxPoint.begin(), boxPoint.end(), pointCompareX);

    int index1, index2, index3, index4;
    if (boxPoint[1].y > boxPoint[0].y) { index1 = 0; index4 = 1; } else { index1 = 1; index4 = 0; }
    if (boxPoint[3].y > boxPoint[2].y) { index2 = 2; index3 = 3; } else { index2 = 3; index3 = 2; }

    std::vector<cv::Point2f> minBox(4);
    minBox[0] = boxPoint[index1];
    minBox[1] = boxPoint[index2];
    minBox[2] = boxPoint[index3];
    minBox[3] = boxPoint[index4];
    return minBox;
}

float boxScoreFast(const std::vector<cv::Point2f>& boxes, const cv::Mat& pred) {
    int width = pred.cols;
    int height = pred.rows;

    float arrayX[4] = {boxes[0].x, boxes[1].x, boxes[2].x, boxes[3].x};
    float arrayY[4] = {boxes[0].y, boxes[1].y, boxes[2].y, boxes[3].y};

    int minX = clampValue(static_cast<int>(std::floor(*std::min_element(arrayX, arrayX + 4))), 0, width - 1);
    int maxX = clampValue(static_cast<int>(std::ceil(*std::max_element(arrayX, arrayX + 4))), 0, width - 1);
    int minY = clampValue(static_cast<int>(std::floor(*std::min_element(arrayY, arrayY + 4))), 0, height - 1);
    int maxY = clampValue(static_cast<int>(std::ceil(*std::max_element(arrayY, arrayY + 4))), 0, height - 1);

    cv::Mat mask = cv::Mat::zeros(maxY - minY + 1, maxX - minX + 1, CV_8UC1);
    cv::Point box[4];
    for (int i = 0; i < 4; ++i) {
        box[i] = cv::Point(static_cast<int>(boxes[i].x) - minX, static_cast<int>(boxes[i].y) - minY);
    }
    const cv::Point* pts[1] = {box};
    int npts[] = {4};
    cv::fillPoly(mask, pts, npts, 1, cv::Scalar(1));

    cv::Mat cropped;
    pred(cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1)).copyTo(cropped);
    return static_cast<float>(cv::mean(cropped, mask)[0]);
}

namespace {
float getContourArea(const std::vector<cv::Point2f>& box, float unClipRatio) {
    size_t size = box.size();
    float area = 0.0f, dist = 0.0f;
    for (size_t i = 0; i < size; i++) {
        area += box[i].x * box[(i + 1) % size].y - box[i].y * box[(i + 1) % size].x;
        dist += std::sqrt(
            (box[i].x - box[(i + 1) % size].x) * (box[i].x - box[(i + 1) % size].x) +
            (box[i].y - box[(i + 1) % size].y) * (box[i].y - box[(i + 1) % size].y));
    }
    area = std::fabs(area / 2.0f);
    return area * unClipRatio / dist;
}
}

cv::RotatedRect unClipBox(std::vector<cv::Point2f> box, float unClipRatio) {
    float distance = getContourArea(box, unClipRatio);

    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(static_cast<int>(box[0].x), static_cast<int>(box[0].y))
      << ClipperLib::IntPoint(static_cast<int>(box[1].x), static_cast<int>(box[1].y))
      << ClipperLib::IntPoint(static_cast<int>(box[2].x), static_cast<int>(box[2].y))
      << ClipperLib::IntPoint(static_cast<int>(box[3].x), static_cast<int>(box[3].y));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point2f> points;
    for (auto& path : soln) {
        for (auto& pt : path) {
            points.emplace_back(static_cast<float>(pt.X), static_cast<float>(pt.Y));
        }
    }
    if (points.empty()) {
        return cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    }
    return cv::minAreaRect(points);
}

std::vector<float> substractMeanNormalize(const cv::Mat& src, const float* meanVals, const float* normVals) {
    size_t numChannels = src.channels();
    size_t imageSize = static_cast<size_t>(src.cols) * src.rows;
    std::vector<float> out(imageSize * numChannels);
    for (size_t pid = 0; pid < imageSize; pid++) {
        for (size_t ch = 0; ch < numChannels; ++ch) {
            float data = static_cast<float>(src.data[pid * numChannels + ch]) * normVals[ch] - meanVals[ch] * normVals[ch];
            out[ch * imageSize + pid] = data;
        }
    }
    return out;
}

cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box) {
    // Guards below exist because this is a public function custom pipelines
    // can call directly with arbitrary boxes (not just ones DBNet produced
    // and already clamped to image bounds) — a degenerate box here used to
    // throw a cv::Exception from warpPerspective/Rect deep inside OpenCV;
    // it now returns an empty Mat instead, so callers can check `.empty()`
    // like every other guard in this codebase.
    if (src.empty() || box.size() != 4) {
        return {};
    }

    cv::Mat image;
    src.copyTo(image);
    std::vector<cv::Point> points = box;

    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    int left = clampValue(*std::min_element(collectX, collectX + 4), 0, image.cols - 1);
    int right = clampValue(*std::max_element(collectX, collectX + 4), 0, image.cols - 1);
    int top = clampValue(*std::min_element(collectY, collectY + 4), 0, image.rows - 1);
    int bottom = clampValue(*std::max_element(collectY, collectY + 4), 0, image.rows - 1);
    if (right <= left || bottom <= top) {
        return {}; // guard: degenerate/zero-area box after clamping to image bounds
    }

    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    for (auto& point : points) {
        point.x -= left;
        point.y -= top;
    }

    int imgCropWidth = static_cast<int>(std::sqrt(
        std::pow(points[0].x - points[1].x, 2) + std::pow(points[0].y - points[1].y, 2)));
    int imgCropHeight = static_cast<int>(std::sqrt(
        std::pow(points[0].x - points[3].x, 2) + std::pow(points[0].y - points[3].y, 2)));
    if (imgCropWidth <= 0 || imgCropHeight <= 0) {
        return {}; // guard: box points collapsed to a line/point, no valid quad to warp
    }

    cv::Point2f ptsDst[4] = {
        {0.f, 0.f}, {static_cast<float>(imgCropWidth), 0.f},
        {static_cast<float>(imgCropWidth), static_cast<float>(imgCropHeight)},
        {0.f, static_cast<float>(imgCropHeight)},
    };
    cv::Point2f ptsSrc[4] = {
        {static_cast<float>(points[0].x), static_cast<float>(points[0].y)},
        {static_cast<float>(points[1].x), static_cast<float>(points[1].y)},
        {static_cast<float>(points[2].x), static_cast<float>(points[2].y)},
        {static_cast<float>(points[3].x), static_cast<float>(points[3].y)},
    };
    cv::Mat m = cv::getPerspectiveTransform(ptsSrc, ptsDst);

    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, m, cv::Size(imgCropWidth, imgCropHeight), cv::BORDER_REPLICATE);

    if (static_cast<float>(partImg.rows) >= static_cast<float>(partImg.cols) * 1.5f) {
        cv::Mat rotated(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, rotated);
        cv::flip(rotated, rotated, 0);
        return rotated;
    }
    return partImg;
}

cv::Mat matRotateClockWise180(cv::Mat src) {
    if (src.empty()) {
        return src; // guard: e.g. a degenerate crop upstream (getRotateCropImage)
    }
    cv::flip(src, src, 0);
    cv::flip(src, src, 1);
    return src;
}

namespace {

cv::Point2f lerpPt(const cv::Point& a, const cv::Point& b, float t) {
    return {
        static_cast<float>(a.x) + t * static_cast<float>(b.x - a.x),
        static_cast<float>(a.y) + t * static_cast<float>(b.y - a.y),
    };
}

std::vector<cv::Point> toCvPoints(const cv::Point2f* p, int n) {
    std::vector<cv::Point> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        out.emplace_back(static_cast<int>(std::lround(p[i].x)),
                         static_cast<int>(std::lround(p[i].y)));
    }
    return out;
}

// Column "ink" = fraction of dark-ish pixels (text on light receipt paper).
std::vector<float> columnInk(const cv::Mat& gray) {
    const int w = gray.cols;
    const int h = gray.rows;
    std::vector<float> ink(static_cast<size_t>(w), 0.f);
    if (w <= 0 || h <= 0) return ink;
    for (int x = 0; x < w; ++x) {
        int dark = 0;
        for (int y = 0; y < h; ++y) {
            if (gray.at<uchar>(y, x) < 160) ++dark; // ponytail: fixed thresh OK for receipts
        }
        ink[static_cast<size_t>(x)] = static_cast<float>(dark) / static_cast<float>(h);
    }
    return ink;
}

// Smooth with small box filter so single-pixel gaps don't win.
void smooth1d(std::vector<float>& v, int radius = 2) {
    if (v.empty() || radius <= 0) return;
    std::vector<float> out(v.size(), 0.f);
    for (size_t i = 0; i < v.size(); ++i) {
        float sum = 0.f;
        int n = 0;
        for (int d = -radius; d <= radius; ++d) {
            int j = static_cast<int>(i) + d;
            if (j < 0 || j >= static_cast<int>(v.size())) continue;
            sum += v[static_cast<size_t>(j)];
            ++n;
        }
        out[i] = n ? sum / static_cast<float>(n) : 0.f;
    }
    v.swap(out);
}

} // namespace

std::vector<RawTextBox> maybeSplitOvermergedBox(const RawTextBox& box,
                                                const cv::Mat& src,
                                                float minAspect,
                                                float minGapDepth) {
    if (src.empty() || box.boxPoint.size() != 4) {
        return {box};
    }

    cv::Mat crop = getRotateCropImage(src, box.boxPoint);
    if (crop.empty() || crop.cols < 24 || crop.rows < 4) {
        return {box};
    }

    const float aspect = static_cast<float>(crop.cols) / static_cast<float>(crop.rows);
    if (aspect < minAspect) {
        return {box};
    }

    cv::Mat gray;
    if (crop.channels() == 1) {
        gray = crop;
    } else {
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    }

    auto ink = columnInk(gray);
    smooth1d(ink, 2);

    // Search valley in middle 50% of width (avoid edge padding).
    const int w = static_cast<int>(ink.size());
    const int lo = w / 4;
    const int hi = (3 * w) / 4;
    if (hi <= lo + 2) return {box};

    float median = 0.f;
    {
        std::vector<float> mid(ink.begin() + lo, ink.begin() + hi);
        std::nth_element(mid.begin(), mid.begin() + mid.size() / 2, mid.end());
        median = mid[mid.size() / 2];
    }
    if (median < 0.05f) return {box}; // almost blank crop

    int bestX = -1;
    float bestInk = 1.f;
    for (int x = lo; x < hi; ++x) {
        if (ink[static_cast<size_t>(x)] < bestInk) {
            bestInk = ink[static_cast<size_t>(x)];
            bestX = x;
        }
    }
    if (bestX < 0) return {box};
    // Gap must be clearly quieter than typical text columns.
    if (bestInk > median * (1.f - minGapDepth)) {
        return {box};
    }
    // Also require absolute quiet-ish valley (not just relative on dense text).
    if (bestInk > 0.22f) {
        return {box};
    }

    const float t = (static_cast<float>(bestX) + 0.5f) / static_cast<float>(w);
    // boxPoint order from getRotateCropImage / DBNet: 0 TL, 1 TR, 2 BR, 3 BL
    const auto& p = box.boxPoint;
    cv::Point2f midTop = lerpPt(p[0], p[1], t);
    cv::Point2f midBot = lerpPt(p[3], p[2], t);

    cv::Point2f leftPts[4] = {
        {static_cast<float>(p[0].x), static_cast<float>(p[0].y)},
        midTop,
        midBot,
        {static_cast<float>(p[3].x), static_cast<float>(p[3].y)},
    };
    cv::Point2f rightPts[4] = {
        midTop,
        {static_cast<float>(p[1].x), static_cast<float>(p[1].y)},
        {static_cast<float>(p[2].x), static_cast<float>(p[2].y)},
        midBot,
    };

    RawTextBox left{toCvPoints(leftPts, 4), box.score};
    RawTextBox right{toCvPoints(rightPts, 4), box.score};
    return {left, right};
}

std::vector<RawTextBox> expandOvermergedBoxes(const std::vector<RawTextBox>& boxes,
                                             const cv::Mat& src) {
    std::vector<RawTextBox> out;
    out.reserve(boxes.size() + boxes.size() / 4);
    for (const auto& b : boxes) {
        auto parts = maybeSplitOvermergedBox(b, src);
        for (auto& p : parts) out.push_back(std::move(p));
    }
    return out;
}

void sortLinesReadingOrder(std::vector<LinePrediction>& lines) {
    auto centroid = [](const Polygon& poly) {
        float x = 0.f, y = 0.f;
        if (poly.empty()) return std::pair<float, float>{0.f, 0.f};
        for (const auto& pt : poly) {
            x += pt.x;
            y += pt.y;
        }
        const float n = static_cast<float>(poly.size());
        return std::pair<float, float>{x / n, y / n};
    };
    std::stable_sort(lines.begin(), lines.end(), [&](const LinePrediction& a, const LinePrediction& b) {
        auto ca = centroid(a.polygon);
        auto cb = centroid(b.polygon);
        // Same visual row if y within ~half a typical line — use 12px fallback.
        const float yTol = 12.f;
        if (std::fabs(ca.second - cb.second) > yTol) {
            return ca.second < cb.second;
        }
        return ca.first < cb.first;
    });
}

} // namespace arbo::ocr
