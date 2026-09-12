// Adapted from RapidAI/RapidOcrOnnx's OcrUtils.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. getScaleParam/getMinBoxes/boxScoreFast/
// unClipBox/substractMeanNormalize/getRotateCropImage/matRotateClockWise180
// are near-verbatim ports of the corresponding upstream functions.
#include "arboOCR/ocr_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib> // std::abs(int) for the fast-path predicate below
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
    // RapidOCR's `Det.limit_type = max` semantic: targetSize is a CEILING on
    // the long side, never a target to grow to. Upscaling a small image is
    // pure waste — the detector pays full O(w*h) cost on interpolated pixels
    // that carry no extra text detail — and arboOCR's default
    // detLimitSideLen=960 was measured as a ceiling on receipts (1536
    // over-merged, 960 recovered ~2 points), so growing past the source
    // resolution runs the model outside the range it was tuned on.
    ratio = std::min(ratio, 1.0f);

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

cv::Mat getRotateCropImage(const cv::Mat& src, std::vector<cv::Point> box, bool* wasTransposed) {
    // Set up front so every early return below reports "not transposed"
    // without each guard having to remember to.
    if (wasTransposed) {
        *wasTransposed = false;
    }
    // Guards below exist because this is a public function custom pipelines
    // can call directly with arbitrary boxes (not just ones DBNet produced
    // and already clamped to image bounds) — a degenerate box here used to
    // throw a cv::Exception from warpPerspective/Rect deep inside OpenCV;
    // it now returns an empty Mat instead, so callers can check `.empty()`
    // like every other guard in this codebase.
    if (src.empty() || box.size() != 4) {
        return {};
    }

    std::vector<cv::Point> points = box;
    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};

    // ROI-first: submat view + copyTo of only the clamped bounding box —
    // callers pass this once per detected box against the full page, and a
    // per-box full-image copy dominated cost for small boxes (oar-ocr crops
    // the ROI up front for the same reason).
    int left = clampValue(*std::min_element(collectX, collectX + 4), 0, src.cols - 1);
    int right = clampValue(*std::max_element(collectX, collectX + 4), 0, src.cols - 1);
    int top = clampValue(*std::min_element(collectY, collectY + 4), 0, src.rows - 1);
    int bottom = clampValue(*std::max_element(collectY, collectY + 4), 0, src.rows - 1);
    if (right <= left || bottom <= top) {
        return {}; // guard: degenerate/zero-area box after clamping to image bounds
    }

    cv::Mat imgCrop;
    src(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);
    const int cropWidth = right - left;
    const int cropHeight = bottom - top;

    for (auto& point : points) {
        point.x -= left;
        point.y -= top;
    }

    // Near-axis-aligned fast path with 2px Chebyshev tolerance per corner.
    // A strict equality could never fire here: boxes arrive as DBNet floats,
    // pass through unClipBox (Clipper jtRound) and minAreaRect (a small
    // non-zero angle), then get truncated to int in detector.cpp findRsBoxes
    // (static_cast<int> on the un-scaled coords) — so the quad is consistently
    // 1-2px skewed against its own clamped ROI bbox, never exactly on it.
    // Safe: the fast path crops the ROI instead of warping, so output differs
    // from the warp by at most those ~2px of edge; the recognizer tolerates a
    // small translation, and genuinely skewed boxes miss the predicate and
    // still take the perspective path below (clipped boxes keep too — their
    // clamped corner coordinate is not the box extent).
    auto nearPt = [](const cv::Point& p, int x, int y) {
        return std::abs(p.x - x) <= 2 && std::abs(p.y - y) <= 2;
    };
    if (nearPt(points[0], 0, 0) && nearPt(points[1], cropWidth, 0) &&
        nearPt(points[2], cropWidth, cropHeight) && nearPt(points[3], 0, cropHeight)) {
        if (static_cast<float>(imgCrop.rows) >= static_cast<float>(imgCrop.cols) * 1.5f) {
            cv::Mat rotated(imgCrop.rows, imgCrop.cols, imgCrop.depth());
            cv::transpose(imgCrop, rotated);
            cv::flip(rotated, rotated, 0);
            if (wasTransposed) {
                *wasTransposed = true;
            }
            return rotated;
        }
        return imgCrop;
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
        if (wasTransposed) {
            *wasTransposed = true;
        }
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

// Same operation on Polygon's point type (arbo::ocr::Point2f, not cv's) —
// spanToPolygon walks the very same quad edges as maybeSplitOvermergedBox,
// just on already-float page coordinates.
Point2f lerpPt(const Point2f& a, const Point2f& b, float t) {
    return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)};
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
    // Same visual row if y within ~half a typical line. A fixed pixel tolerance
    // is resolution-dependent — 12px is about half a line on a phone photo, a
    // small fraction of one on a 300 DPI scan (rows fragment into single
    // fields) and more than a whole line on a thumbnail (rows merge) — so
    // derive it from the median polygon height instead. Half a median line
    // height is the widest margin available on both sides: within a row,
    // neighbouring fields jitter by well under half a line, while the next row
    // sits at least a full line away. Scale-invariant by construction: the
    // same page at 1x and 4x yields the same ordering.
    constexpr float kRowTolFraction = 0.5f;
    constexpr float kMinRowTol = 2.f; // floor for degenerate (zero-height) polygons
    float yTol = kMinRowTol;
    std::vector<float> heights;
    heights.reserve(lines.size());
    for (const auto& line : lines) {
        if (line.polygon.empty()) continue;
        float minY = line.polygon.front().y;
        float maxY = minY;
        for (const auto& pt : line.polygon) {
            minY = std::min(minY, pt.y);
            maxY = std::max(maxY, pt.y);
        }
        heights.push_back(maxY - minY);
    }
    if (!heights.empty()) {
        std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
        yTol = std::max(kMinRowTol, kRowTolFraction * heights[heights.size() / 2]);
    }

    std::stable_sort(lines.begin(), lines.end(), [&](const LinePrediction& a, const LinePrediction& b) {
        auto ca = centroid(a.polygon);
        auto cb = centroid(b.polygon);
        if (std::fabs(ca.second - cb.second) > yTol) {
            return ca.second < cb.second;
        }
        return ca.first < cb.first;
    });
}

namespace {

// 0=letter-ish, 1=digit, 2=other — ASCII-first; multi-byte → letter-ish.
int tokenClass(const std::string& tok) {
    if (tok.empty()) return 2;
    unsigned char c = static_cast<unsigned char>(tok[0]);
    if (c >= '0' && c <= '9') return 1;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return 0;
    if (c >= 0x80) return 0; // ponytail: treat UTF-8 lead as letter-ish
    return 2;
}

bool isSpaceToken(const std::string& t) {
    return t == " " || t == "\t";
}

// Decode next UTF-8 codepoint; advance i. Returns 0 on invalid/truncated.
uint32_t nextCp(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        ++i;
        return c;
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = (c & 0x1F) << 6;
        cp |= static_cast<unsigned char>(s[i + 1]) & 0x3F;
        i += 2;
        return cp;
    }
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = (c & 0x0F) << 12;
        cp |= (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6;
        cp |= static_cast<unsigned char>(s[i + 2]) & 0x3F;
        i += 3;
        return cp;
    }
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = (c & 0x07) << 18;
        cp |= (static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12;
        cp |= (static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6;
        cp |= static_cast<unsigned char>(s[i + 3]) & 0x3F;
        i += 4;
        return cp;
    }
    ++i;
    return 0;
}

bool hasCjk(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = nextCp(s, i);
        // CJK unified + kana + hangul + CJK compat (same idea as ppu)
        if ((cp >= 0x2E80 && cp <= 0x9FFF) ||
            (cp >= 0xAC00 && cp <= 0xD7AF) ||
            (cp >= 0xF900 && cp <= 0xFAFF)) {
            return true;
        }
    }
    return false;
}

bool hasLetterOrDigit(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = nextCp(s, i);
        if ((cp >= '0' && cp <= '9') ||
            (cp >= 'A' && cp <= 'Z') ||
            (cp >= 'a' && cp <= 'z') ||
            (cp >= 0xC0 && cp <= 0x24F) || // Latin extended (rough)
            (cp >= 0x2E80 && cp <= 0x9FFF) ||
            (cp >= 0xAC00 && cp <= 0xD7AF)) {
            return true;
        }
    }
    return false;
}

} // namespace

void injectGapSpaces(std::vector<std::string>& tokens,
                     std::vector<float>& positions,
                     std::vector<float>* scores,
                     std::vector<TokenSpan>* spans) {
    if (tokens.size() < 4 || positions.size() != tokens.size()) return;
    if (scores && scores->size() != tokens.size()) return;
    if (spans && spans->size() != tokens.size()) return;

    std::vector<float> deltas;
    deltas.reserve(positions.size());
    for (size_t i = 1; i < positions.size(); ++i) {
        deltas.push_back(positions[i] - positions[i - 1]);
    }
    std::vector<float> sorted = deltas;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[sorted.size() / 2];
    if (median <= 0.f) return;
    float quantum = 0.f;
    for (float d : sorted) {
        if (d > 0.f) {
            quantum = d;
            break;
        }
    }
    if (quantum <= 0.f) return;

    constexpr float kCross = 1.5f;
    constexpr float kSame = 2.5f;

    for (size_t i = tokens.size(); i-- > 1;) {
        const float prev = positions[i - 1];
        const float curr = positions[i];
        const float k = (tokenClass(tokens[i]) == tokenClass(tokens[i - 1])) ? kSame : kCross;
        if (curr - prev > median + k * quantum &&
            !isSpaceToken(tokens[i]) &&
            !isSpaceToken(tokens[i - 1]) &&
            tokens[i] != tokens[i - 1]) {
            tokens.insert(tokens.begin() + static_cast<std::ptrdiff_t>(i), " ");
            positions.insert(positions.begin() + static_cast<std::ptrdiff_t>(i), (prev + curr) * 0.5f);
            if (scores) {
                const float sc = ((*scores)[i - 1] + (*scores)[i]) * 0.5f;
                scores->insert(scores->begin() + static_cast<std::ptrdiff_t>(i), sc);
            }
            if (spans) {
                // The natural extent of an injected space is the gap it stands
                // for: from where the left glyph ended to where the right one
                // begins. Read the neighbours before inserting — the loop runs
                // backwards precisely so indices below i stay valid. min/max
                // because CTC spans of adjacent tokens can overlap by a hair,
                // and TokenSpan documents begin <= end.
                const float gapBegin = (*spans)[i - 1].end;
                const float gapEnd = (*spans)[i].begin;
                spans->insert(spans->begin() + static_cast<std::ptrdiff_t>(i),
                              TokenSpan{std::min(gapBegin, gapEnd), std::max(gapBegin, gapEnd)});
            }
        }
    }
}

Polygon spanToPolygon(const Polygon& lineQuad, TokenSpan span,
                      bool wasTransposed, bool wasRotated180) {
    if (lineQuad.size() != 4) {
        return {}; // guard: not a quad, no reading axis to interpolate along
    }

    float begin = span.begin;
    float end = span.end;
    if (begin > end) std::swap(begin, end);
    begin = clampValue(begin, 0.0f, 1.0f);
    end = clampValue(end, 0.0f, 1.0f);

    // Undoing the 180 flip is a change of variable on the fraction alone.
    // matRotateClockWise180 is applied to the finished crop (engine.cpp), so
    // final(r,c) == crop(R-1-r, C-1-c): the reading axis is reversed but it is
    // still the *same* axis of the quad. Hence f -> 1-f, i.e. [b,e] -> [1-e,1-b],
    // and this one substitution covers both the plain and the transposed case
    // below — which is why the composition needs no extra branch.
    if (wasRotated180) {
        const float flippedBegin = 1.0f - end;
        end = 1.0f - begin;
        begin = flippedBegin;
    }

    // Interpolating along the quad's edges is EXACT here, not an approximation.
    // getRotateCropImage warps the quad onto the crop rectangle, and that warp
    // is affine along the reading axis iff the quad's top and bottom edges are
    // parallel. They are: the quad is born a rectangle
    // (cv::minAreaRect -> unClipBox -> cv::minAreaRect in detector.cpp) and the
    // only thing applied afterwards is the anisotropic x/ratioWidth,
    // y/ratioHeight rescale at detector.cpp:49-50 — an affine map, which
    // preserves parallelism. So a lerp along the top and bottom edges lands
    // exactly where the recognizer saw the token. The one residual is the
    // integer truncation at detector.cpp:51-53 (<=1px per vertex, ~0.1-0.4px
    // median); it is purely tangential — it slides a word box a fraction of a
    // pixel along its own line, never off it.
    const auto& p = lineQuad;

    if (wasTransposed) {
        // getRotateCropImage's tall-box branch does transpose + vertical flip,
        // i.e. crop(r,c) == partImg(c, W-1-r) with partImg H rows x W cols.
        // Read that off: the crop's left-to-right axis c is partImg's y, so the
        // recognizer reads TOP-TO-BOTTOM down the quad and f is a fraction of
        // the quad's height; a span therefore covers the quad's full width.
        // (The crop's own vertical axis r maps to x = W-1-r, i.e. the crop's top
        // row is the quad's right edge — irrelevant to the span, which spans all
        // of x, but it is what makes the rotation 90 degrees CCW and not CW.)
        // In the quad's frame the band's corners are (u,v) = (0,b) (1,b) (1,e)
        // (0,e), which is TL,TR,BR,BL in page order because getMinBoxes already
        // put p0 at the page top-left.
        return {
            lerpPt(p[0], p[3], begin), // left edge at begin
            lerpPt(p[1], p[2], begin), // right edge at begin
            lerpPt(p[1], p[2], end),
            lerpPt(p[0], p[3], end),
        };
    }

    // Base case: f runs left-to-right along the quad, exactly the operation
    // maybeSplitOvermergedBox performs to cut a box in two.
    return {
        lerpPt(p[0], p[1], begin),
        lerpPt(p[0], p[1], end),
        lerpPt(p[3], p[2], end),
        lerpPt(p[3], p[2], begin),
    };
}

std::vector<WordBox> groupTokensIntoWords(const std::vector<std::string>& tokens,
                                          const std::vector<TokenSpan>& spans,
                                          const std::vector<float>& scores,
                                          const Polygon& lineQuad,
                                          bool wasTransposed,
                                          bool wasRotated180) {
    std::vector<WordBox> words;
    if (tokens.size() != spans.size() || tokens.size() != scores.size()) {
        return words; // guard: index-aligned or nothing — never read past the shortest
    }
    words.reserve(tokens.size() / 4 + 1);

    std::string text;
    TokenSpan span{};
    float scoreSum = 0.0f;
    size_t scoreCount = 0;

    auto flush = [&]() {
        if (scoreCount > 0 && !text.empty()) {
            words.push_back(WordBox{
                spanToPolygon(lineQuad, span, wasTransposed, wasRotated180),
                text,
                scoreSum / static_cast<float>(scoreCount),
            });
        }
        text.clear();
        scoreSum = 0.0f;
        scoreCount = 0;
    };

    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& tok = tokens[i];
        if (tok.empty()) continue;
        if (isSpaceToken(tok)) {
            flush(); // the space ends the word and is not a word itself
            continue;
        }
        if (hasCjk(tok)) {
            // CJK is not space-delimited, so every character is its own word —
            // same rule as RapidOCR's return_word_box. Reusing the whole-string
            // hasCjk is fine on a single token: one token is one character.
            flush();
            words.push_back(WordBox{
                spanToPolygon(lineQuad, spans[i], wasTransposed, wasRotated180),
                tok,
                scores[i],
            });
            continue;
        }
        if (scoreCount == 0) {
            span.begin = spans[i].begin;
        }
        span.end = spans[i].end;
        text += tok;
        scoreSum += scores[i];
        ++scoreCount;
    }
    flush();
    return words;
}

void refineDecodedText(std::string& text) {
    const bool cjk = hasCjk(text);
    std::string out;
    out.reserve(text.size());
    bool prevSpace = false;
    for (size_t i = 0; i < text.size();) {
        size_t j = i;
        uint32_t cp = nextCp(text, j);
        if (cp == ' ' || cp == 0x3000) {
            if (!prevSpace) out.push_back(' ');
            prevSpace = true;
            i = j;
            continue;
        }
        prevSpace = false;
        if (!cjk && cp >= 0xFF01 && cp <= 0xFF5E) {
            out.push_back(static_cast<char>(cp - 0xFEE0));
        } else {
            out.append(text, i, j - i);
        }
        i = j;
    }
    text.swap(out);
}

bool keepByConfidence(const std::string& text, float confidence, float minimumConfidence) {
    if (minimumConfidence <= 0.f) return true;
    float bar = minimumConfidence;
    if (!hasLetterOrDigit(text)) {
        bar = std::min(1.f, minimumConfidence + 0.3f);
    }
    return confidence >= bar;
}

} // namespace arbo::ocr
