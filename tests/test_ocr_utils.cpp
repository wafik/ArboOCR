#include <algorithm>

#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include "arboOCR/ocr_utils.hpp"

using namespace arbo::ocr;

TEST_CASE("getScaleParam rounds dims down to a multiple of 32") {
    cv::Mat img(400, 600, CV_8UC3); // 600 wide x 400 tall
    ScaleParam s = getScaleParam(img, 736);
    CHECK(s.srcWidth == 600);
    CHECK(s.srcHeight == 400);
    // 600 is already under the 736 ceiling, so both dims only floor to /32.
    CHECK(s.dstWidth % 32 == 0);
    CHECK(s.dstHeight % 32 == 0);
    CHECK(s.dstWidth > 0);
    CHECK(s.dstHeight > 0);
}

TEST_CASE("getScaleParam never upscales a small image (targetSize is a ceiling)") {
    cv::Mat thumb(150, 200, CV_8UC3); // 200 wide x 150 tall, far under the limit
    ScaleParam s = getScaleParam(thumb, 960);
    CHECK(s.srcWidth == 200);
    CHECK(s.srcHeight == 150);
    // Old behaviour scaled the long side up to 960 and paid detector cost on
    // interpolated pixels; the ceiling semantic only floors to /32.
    CHECK(s.dstWidth == 192);
    CHECK(s.dstHeight == 128);
    CHECK(s.dstWidth <= s.srcWidth);
    CHECK(s.dstHeight <= s.srcHeight);
    CHECK(s.ratioWidth <= 1.0f);
    CHECK(s.ratioHeight <= 1.0f);
}

TEST_CASE("getScaleParam still downscales a large image to the ceiling") {
    cv::Mat page(1080, 1920, CV_8UC3); // 1920 wide x 1080 tall
    ScaleParam s = getScaleParam(page, 960);
    CHECK(s.srcWidth == 1920);
    CHECK(s.srcHeight == 1080);
    CHECK(s.dstWidth == 960);  // long side exactly at the ceiling
    CHECK(s.dstHeight == 512); // 540 floored to a multiple of 32
    CHECK(s.dstWidth % 32 == 0);
    CHECK(s.dstHeight % 32 == 0);
    CHECK(std::max(s.dstWidth, s.dstHeight) <= 960);
}

TEST_CASE("getMinBoxes returns 4 ordered points and correct maxSideLen") {
    cv::RotatedRect rect(cv::Point2f(50, 50), cv::Size2f(100, 40), 0.0f);
    float maxSideLen = 0.0f;
    auto box = getMinBoxes(rect, maxSideLen);
    REQUIRE(box.size() == 4);
    CHECK(maxSideLen == doctest::Approx(100.0f));
}

TEST_CASE("boxScoreFast returns higher score for a bright region under the box") {
    cv::Mat pred = cv::Mat::zeros(100, 100, CV_32F);
    pred(cv::Rect(10, 10, 30, 30)).setTo(0.9f); // bright square
    std::vector<cv::Point2f> box = {
        {10, 10}, {40, 10}, {40, 40}, {10, 40}
    };
    float score = boxScoreFast(box, pred);
    CHECK(score > 0.8f);
}

TEST_CASE("unClipBox expands a box outward (result area >= input area)") {
    std::vector<cv::Point2f> box = {
        {0, 0}, {100, 0}, {100, 40}, {0, 40}
    };
    cv::RotatedRect result = unClipBox(box, 1.6f);
    CHECK(result.size.width * result.size.height >= 100.0f * 40.0f);
}

TEST_CASE("substractMeanNormalize produces correct output length") {
    cv::Mat img = cv::Mat::ones(10, 20, CV_8UC3) * 128; // 20 wide, 10 tall, 3 channels
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float norm[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    auto values = substractMeanNormalize(img, mean, norm);
    CHECK(values.size() == 10 * 20 * 3);
}

TEST_CASE("matRotateClockWise180 flips both axes") {
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    img.at<uchar>(0, 0) = 255; // top-left corner marked
    cv::Mat rotated = matRotateClockWise180(img.clone());
    CHECK(rotated.at<uchar>(9, 9) == 255); // should land at bottom-right
}

TEST_CASE("matRotateClockWise180 on an empty Mat returns empty, does not throw") {
    cv::Mat empty;
    CHECK_NOTHROW(matRotateClockWise180(empty).empty());
}

TEST_CASE("getRotateCropImage returns empty Mat for a degenerate (zero-area) box, does not throw") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    // All 4 points identical -> zero width/height after the crop math.
    std::vector<cv::Point> degenerateBox = {{10, 10}, {10, 10}, {10, 10}, {10, 10}};
    cv::Mat result;
    CHECK_NOTHROW(result = getRotateCropImage(src, degenerateBox));
    CHECK(result.empty());
}

TEST_CASE("getRotateCropImage returns empty Mat for a box outside image bounds, does not throw") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    std::vector<cv::Point> outOfBoundsBox = {{200, 200}, {250, 200}, {250, 250}, {200, 250}};
    cv::Mat result;
    CHECK_NOTHROW(result = getRotateCropImage(src, outOfBoundsBox));
    CHECK(result.empty());
}

TEST_CASE("maybeSplitOvermergedBox splits wide crop with ink gap") {
    // White background, two dark text blocks with a clear gap in the middle.
    cv::Mat src(40, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(src, cv::Rect(5, 5, 70, 30), cv::Scalar(20, 20, 20), cv::FILLED);
    cv::rectangle(src, cv::Rect(125, 5, 70, 30), cv::Scalar(20, 20, 20), cv::FILLED);
    RawTextBox box;
    box.boxPoint = {{0, 0}, {200, 0}, {200, 40}, {0, 40}};
    box.score = 0.9f;
    auto parts = maybeSplitOvermergedBox(box, src, /*minAspect=*/3.0f, /*minGapDepth=*/0.3f);
    REQUIRE(parts.size() == 2);
    CHECK(parts[0].score == doctest::Approx(0.9f));
    CHECK(parts[1].score == doctest::Approx(0.9f));
}

TEST_CASE("maybeSplitOvermergedBox keeps short box intact") {
    cv::Mat src(40, 60, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(src, cv::Rect(5, 5, 50, 30), cv::Scalar(20, 20, 20), cv::FILLED);
    RawTextBox box;
    box.boxPoint = {{0, 0}, {60, 0}, {60, 40}, {0, 40}};
    box.score = 0.8f;
    auto parts = maybeSplitOvermergedBox(box, src);
    REQUIRE(parts.size() == 1);
}

TEST_CASE("sortLinesReadingOrder sorts by y then x") {
    LinePrediction a, b, c;
    a.polygon = {{100, 10}, {140, 10}, {140, 20}, {100, 20}};
    a.text = "right-top";
    b.polygon = {{10, 10}, {50, 10}, {50, 20}, {10, 20}};
    b.text = "left-top";
    c.polygon = {{10, 40}, {50, 40}, {50, 50}, {10, 50}};
    c.text = "bottom";
    std::vector<LinePrediction> lines = {a, c, b};
    sortLinesReadingOrder(lines);
    REQUIRE(lines.size() == 3);
    CHECK(lines[0].text == "left-top");
    CHECK(lines[1].text == "right-top");
    CHECK(lines[2].text == "bottom");
}

TEST_CASE("sortLinesReadingOrder gives the same order at 1x and 4x coordinates") {
    // Same logical page at two resolutions. The top row's two columns sit 4px
    // apart in y at 1x — 16px at 4x, which is past the old hard-coded 12px
    // tolerance, so the high-DPI copy used to fragment that row and emit the
    // right-hand column first. With a tolerance derived from median line
    // height the ordering is identical at both scales.
    auto buildPage = [](float k) {
        auto rect = [k](float x0, float y0, float x1, float y1, const char* text) {
            LinePrediction line;
            line.polygon = {{x0 * k, y0 * k}, {x1 * k, y0 * k}, {x1 * k, y1 * k}, {x0 * k, y1 * k}};
            line.text = text;
            return line;
        };
        // Deliberately shuffled input order.
        return std::vector<LinePrediction>{
            rect(100, 6, 140, 20, "row0-right"),
            rect(10, 70, 50, 84, "row2-left"),
            rect(10, 10, 50, 24, "row0-left"),
            rect(100, 41, 140, 55, "row1-right"),
            rect(10, 40, 50, 54, "row1-left"),
        };
    };
    const std::vector<std::string> expected = {
        "row0-left", "row0-right", "row1-left", "row1-right", "row2-left"};

    for (float k : {1.0f, 4.0f}) {
        CAPTURE(k);
        std::vector<LinePrediction> lines = buildPage(k);
        sortLinesReadingOrder(lines);
        REQUIRE(lines.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            CHECK(lines[i].text == expected[i]);
        }
    }
}

TEST_CASE("sortLinesReadingOrder handles empty, single and zero-height input") {
    std::vector<LinePrediction> none;
    CHECK_NOTHROW(sortLinesReadingOrder(none));
    CHECK(none.empty());

    LinePrediction only;
    only.polygon = {{10, 10}, {50, 10}, {50, 24}, {10, 24}};
    only.text = "solo";
    std::vector<LinePrediction> one = {only};
    CHECK_NOTHROW(sortLinesReadingOrder(one));
    REQUIRE(one.size() == 1);
    CHECK(one[0].text == "solo");

    // Degenerate: zero-height polygons (median height 0 -> tolerance floor)
    // plus an empty polygon (centroid 0,0). Must still order top-to-bottom.
    LinePrediction flatLow, flatHigh, noPoly;
    flatLow.polygon = {{10, 60}, {50, 60}, {50, 60}, {10, 60}};
    flatLow.text = "low";
    flatHigh.polygon = {{10, 20}, {50, 20}, {50, 20}, {10, 20}};
    flatHigh.text = "high";
    noPoly.text = "no-polygon";
    std::vector<LinePrediction> degenerate = {flatLow, noPoly, flatHigh};
    CHECK_NOTHROW(sortLinesReadingOrder(degenerate));
    REQUIRE(degenerate.size() == 3);
    CHECK(degenerate[0].text == "no-polygon");
    CHECK(degenerate[1].text == "high");
    CHECK(degenerate[2].text == "low");
}

TEST_CASE("injectGapSpaces inserts space on wide cross-class gap") {
    // positions with a clear column gap between letter and digit clusters
    std::vector<std::string> tokens = {"A", "B", "C", "D", "1", "2", "3", "4"};
    std::vector<float> pos = {0.05f, 0.10f, 0.15f, 0.20f, 0.70f, 0.75f, 0.80f, 0.85f};
    std::vector<float> scores(8, 0.9f);
    injectGapSpaces(tokens, pos, &scores);
    bool foundSpace = false;
    for (const auto& t : tokens) if (t == " ") foundSpace = true;
    CHECK(foundSpace);
    CHECK(tokens.size() == scores.size());
    CHECK(tokens.size() == pos.size());
}

TEST_CASE("refineDecodedText maps fullwidth colon and collapses spaces") {
    // U+FF1A fullwidth colon = UTF-8 EF BC 9A (avoid \x in string literal for MSVC)
    std::string s = "Tel";
    s += static_cast<char>(0xEF);
    s += static_cast<char>(0xBC);
    s += static_cast<char>(0x9A);
    s += "07  88";
    refineDecodedText(s);
    CHECK(s == "Tel:07 88");
}

TEST_CASE("keepByConfidence matches ppu dual bar") {
    CHECK(keepByConfidence("TOTAL", 0.6f, 0.5f));
    CHECK_FALSE(keepByConfidence("TOTAL", 0.4f, 0.5f));
    CHECK(keepByConfidence("+-", 0.85f, 0.5f)); // symbol bar 0.8
    CHECK_FALSE(keepByConfidence("+-", 0.7f, 0.5f));
    CHECK(keepByConfidence("x", 0.1f, 0.0f)); // disabled
}

TEST_CASE("getRotateCropImage returns empty Mat for an empty source image, does not throw") {
    cv::Mat empty;
    std::vector<cv::Point> box = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    cv::Mat result;
    CHECK_NOTHROW(result = getRotateCropImage(empty, box));
    CHECK(result.empty());
}

TEST_CASE("getRotateCropImage still crops a valid box correctly (regression guard)") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    src(cv::Rect(10, 10, 30, 30)).setTo(cv::Scalar(255, 255, 255));
    std::vector<cv::Point> box = {{10, 10}, {40, 10}, {40, 40}, {10, 40}};
    cv::Mat result = getRotateCropImage(src, box);
    CHECK_FALSE(result.empty());
    CHECK(result.cols > 0);
    CHECK(result.rows > 0);
}

TEST_CASE("getRotateCropImage reports whether it transposed the crop") {
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    src(cv::Rect(5, 5, 90, 90)).setTo(cv::Scalar(255, 255, 255));

    // Wide box (30x30, aspect 1.0): under the rows >= cols*1.5 threshold.
    std::vector<cv::Point> wide = {{10, 10}, {40, 10}, {40, 40}, {10, 40}};
    bool transposed = true; // seeded wrong on purpose: the call must overwrite it
    cv::Mat wideCrop = getRotateCropImage(src, wide, &transposed);
    CHECK_FALSE(wideCrop.empty());
    CHECK_FALSE(transposed);

    // Tall box (30 wide x 80 tall): 80 >= 30*1.5, so it is rotated 90 CCW and
    // the crop comes back wider than it is tall.
    std::vector<cv::Point> tall = {{10, 10}, {40, 10}, {40, 90}, {10, 90}};
    transposed = false;
    cv::Mat tallCrop = getRotateCropImage(src, tall, &transposed);
    CHECK_FALSE(tallCrop.empty());
    CHECK(transposed);
    CHECK(tallCrop.cols > tallCrop.rows);

    // Early-return guards must still leave the flag defined and false.
    cv::Mat empty;
    transposed = true;
    cv::Mat none = getRotateCropImage(empty, tall, &transposed);
    CHECK(none.empty());
    CHECK_FALSE(transposed);

    // Still callable without the flag (default argument).
    CHECK_FALSE(getRotateCropImage(src, wide).empty());
}

namespace {
// 100 wide x 40 tall axis-aligned quad in getMinBoxes order: TL, TR, BR, BL.
Polygon makeQuad(float scale = 1.0f) {
    return {
        {10.0f * scale, 20.0f * scale},
        {110.0f * scale, 20.0f * scale},
        {110.0f * scale, 60.0f * scale},
        {10.0f * scale, 60.0f * scale},
    };
}

void checkPoint(const Point2f& got, float x, float y) {
    CHECK(got.x == doctest::Approx(x));
    CHECK(got.y == doctest::Approx(y));
}
} // namespace

TEST_CASE("spanToPolygon reproduces the quad for a full span and halves it for [0,0.5]") {
    const Polygon quad = makeQuad();

    Polygon full = spanToPolygon(quad, TokenSpan{0.0f, 1.0f}, false, false);
    REQUIRE(full.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        checkPoint(full[i], quad[i].x, quad[i].y);
    }

    // Left half: x runs 10..60, full height, still TL,TR,BR,BL.
    Polygon left = spanToPolygon(quad, TokenSpan{0.0f, 0.5f}, false, false);
    REQUIRE(left.size() == 4);
    checkPoint(left[0], 10.0f, 20.0f);
    checkPoint(left[1], 60.0f, 20.0f);
    checkPoint(left[2], 60.0f, 60.0f);
    checkPoint(left[3], 10.0f, 60.0f);
}

TEST_CASE("spanToPolygon is scale-invariant") {
    const Polygon quad = makeQuad();
    const Polygon quad4x = makeQuad(4.0f);
    const TokenSpan span{0.25f, 0.5f};

    Polygon a = spanToPolygon(quad, span, false, false);
    Polygon b = spanToPolygon(quad4x, span, false, false);
    REQUIRE(a.size() == 4);
    REQUIRE(b.size() == 4);
    // The same span on the same page at 4x resolution is the same box at 4x.
    for (size_t i = 0; i < 4; ++i) {
        checkPoint(b[i], a[i].x * 4.0f, a[i].y * 4.0f);
    }
}

TEST_CASE("spanToPolygon maps a 180-rotated crop's left onto the page's right") {
    const Polygon quad = makeQuad();
    // The crop was flipped 180, so its first quarter is the page's LAST
    // quarter: span [0,0.25] -> [0.75,1] -> x in 85..110.
    Polygon first = spanToPolygon(quad, TokenSpan{0.0f, 0.25f}, false, true);
    REQUIRE(first.size() == 4);
    checkPoint(first[0], 85.0f, 20.0f);
    checkPoint(first[1], 110.0f, 20.0f);
    checkPoint(first[2], 110.0f, 60.0f);
    checkPoint(first[3], 85.0f, 60.0f);

    // And a full span is unchanged by the flip: [0,1] -> [0,1].
    Polygon full = spanToPolygon(quad, TokenSpan{0.0f, 1.0f}, false, true);
    REQUIRE(full.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        checkPoint(full[i], quad[i].x, quad[i].y);
    }
}

TEST_CASE("spanToPolygon reads down the quad when the crop was transposed") {
    const Polygon quad = makeQuad();
    // Transposed crop: the recognizer read top-to-bottom, so the first quarter
    // is the quad's TOP quarter (y 20..30) spanning its full width (x 10..110).
    Polygon top = spanToPolygon(quad, TokenSpan{0.0f, 0.25f}, true, false);
    REQUIRE(top.size() == 4);
    checkPoint(top[0], 10.0f, 20.0f);
    checkPoint(top[1], 110.0f, 20.0f);
    checkPoint(top[2], 110.0f, 30.0f);
    checkPoint(top[3], 10.0f, 30.0f);

    // Transposed AND flipped 180: same vertical reading axis, reversed, so the
    // first quarter lands on the quad's BOTTOM quarter (y 50..60).
    Polygon bottom = spanToPolygon(quad, TokenSpan{0.0f, 0.25f}, true, true);
    REQUIRE(bottom.size() == 4);
    checkPoint(bottom[0], 10.0f, 50.0f);
    checkPoint(bottom[1], 110.0f, 50.0f);
    checkPoint(bottom[2], 110.0f, 60.0f);
    checkPoint(bottom[3], 10.0f, 60.0f);
}

TEST_CASE("spanToPolygon handles degenerate quads and out-of-range spans") {
    // Not a quad -> no reading axis -> empty, not a crash.
    Polygon threePoints = {{0.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 10.0f}};
    CHECK(spanToPolygon(threePoints, TokenSpan{0.0f, 1.0f}, false, false).empty());
    CHECK(spanToPolygon(Polygon{}, TokenSpan{0.0f, 1.0f}, false, false).empty());

    const Polygon quad = makeQuad();
    const Polygon expected = spanToPolygon(quad, TokenSpan{0.0f, 0.5f}, false, false);
    REQUIRE(expected.size() == 4);

    // begin > end is swapped, not read backwards.
    Polygon reversed = spanToPolygon(quad, TokenSpan{0.5f, 0.0f}, false, false);
    REQUIRE(reversed.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        checkPoint(reversed[i], expected[i].x, expected[i].y);
    }

    // Out of [0,1] is clamped, so the box never escapes the line's own quad.
    Polygon clamped = spanToPolygon(quad, TokenSpan{-3.0f, 0.5f}, false, false);
    REQUIRE(clamped.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        checkPoint(clamped[i], expected[i].x, expected[i].y);
    }
    Polygon wide = spanToPolygon(quad, TokenSpan{-1.0f, 7.0f}, false, false);
    REQUIRE(wide.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        checkPoint(wide[i], quad[i].x, quad[i].y);
    }
}

TEST_CASE("groupTokensIntoWords splits on spaces and averages scores") {
    const Polygon quad = makeQuad(); // x 10..110
    std::vector<std::string> tokens = {"h", "i", " ", "y", "o", "u"};
    std::vector<TokenSpan> spans = {
        {0.0f, 0.1f}, {0.1f, 0.2f}, {0.2f, 0.3f},
        {0.3f, 0.4f}, {0.4f, 0.5f}, {0.5f, 0.6f},
    };
    std::vector<float> scores = {0.9f, 0.8f, 0.5f, 0.6f, 0.7f, 0.8f};

    auto words = groupTokensIntoWords(tokens, spans, scores, quad, false, false);
    REQUIRE(words.size() == 2);

    CHECK(words[0].text == "hi");
    CHECK(words[0].score == doctest::Approx(0.85f));
    REQUIRE(words[0].polygon.size() == 4);
    // span [0.0,0.2] of a 100px-wide quad starting at x=10.
    checkPoint(words[0].polygon[0], 10.0f, 20.0f);
    checkPoint(words[0].polygon[1], 30.0f, 20.0f);

    CHECK(words[1].text == "you");
    CHECK(words[1].score == doctest::Approx((0.6f + 0.7f + 0.8f) / 3.0f));
    REQUIRE(words[1].polygon.size() == 4);
    // span [0.3,0.6].
    checkPoint(words[1].polygon[0], 40.0f, 20.0f);
    checkPoint(words[1].polygon[1], 70.0f, 20.0f);
}

TEST_CASE("groupTokensIntoWords gives each CJK character its own word") {
    const Polygon quad = makeQuad();
    // U+4E2D = UTF-8 E4 B8 AD (avoid \x in string literal for MSVC)
    std::string cjk;
    cjk += static_cast<char>(0xE4);
    cjk += static_cast<char>(0xB8);
    cjk += static_cast<char>(0xAD);

    std::vector<std::string> tokens = {"a", cjk, "b"};
    std::vector<TokenSpan> spans = {{0.0f, 0.3f}, {0.3f, 0.6f}, {0.6f, 0.9f}};
    std::vector<float> scores = {0.5f, 0.6f, 0.7f};

    auto words = groupTokensIntoWords(tokens, spans, scores, quad, false, false);
    // The CJK token has no space around it but must still break the run.
    REQUIRE(words.size() == 3);
    CHECK(words[0].text == "a");
    CHECK(words[1].text == cjk);
    CHECK(words[1].score == doctest::Approx(0.6f));
    CHECK(words[2].text == "b");
    REQUIRE(words[1].polygon.size() == 4);
    checkPoint(words[1].polygon[0], 40.0f, 20.0f); // 10 + 0.3*100
    checkPoint(words[1].polygon[1], 70.0f, 20.0f); // 10 + 0.6*100
}

TEST_CASE("groupTokensIntoWords returns empty on mismatched input sizes") {
    const Polygon quad = makeQuad();
    std::vector<std::string> tokens = {"a", "b"};
    std::vector<TokenSpan> spans = {{0.0f, 0.5f}};
    std::vector<float> scores = {0.9f, 0.9f};
    CHECK(groupTokensIntoWords(tokens, spans, scores, quad, false, false).empty());

    std::vector<TokenSpan> spans2 = {{0.0f, 0.5f}, {0.5f, 1.0f}};
    std::vector<float> shortScores = {0.9f};
    CHECK(groupTokensIntoWords(tokens, spans2, shortScores, quad, false, false).empty());

    // Empty in, empty out — not a mismatch.
    std::vector<std::string> noTokens;
    std::vector<TokenSpan> noSpans;
    std::vector<float> noScores;
    CHECK(groupTokensIntoWords(noTokens, noSpans, noScores, quad, false, false).empty());

    // Empty tokens are skipped, not emitted as zero-width words.
    std::vector<std::string> withEmpty = {"a", "", "b"};
    std::vector<TokenSpan> spans3 = {{0.0f, 0.3f}, {0.3f, 0.3f}, {0.3f, 0.6f}};
    std::vector<float> scores3 = {0.9f, 0.1f, 0.9f};
    auto words = groupTokensIntoWords(withEmpty, spans3, scores3, quad, false, false);
    REQUIRE(words.size() == 1);
    CHECK(words[0].text == "ab");
    CHECK(words[0].score == doctest::Approx(0.9f));
}

TEST_CASE("injectGapSpaces keeps an optional spans vector aligned") {
    std::vector<std::string> tokens = {"A", "B", "C", "D", "1", "2", "3", "4"};
    std::vector<float> pos = {0.05f, 0.10f, 0.15f, 0.20f, 0.70f, 0.75f, 0.80f, 0.85f};
    std::vector<float> scores(8, 0.9f);
    std::vector<TokenSpan> spans;
    for (float p : pos) spans.push_back(TokenSpan{p - 0.02f, p + 0.02f});

    injectGapSpaces(tokens, pos, &scores, &spans);

    REQUIRE(tokens.size() == 9); // exactly one space, in the D|1 gap
    CHECK(pos.size() == tokens.size());
    CHECK(scores.size() == tokens.size());
    CHECK(spans.size() == tokens.size());
    REQUIRE(tokens[4] == " ");
    // The injected space's span is the gap itself: previous token's end to
    // next token's begin, i.e. 0.20+0.02 .. 0.70-0.02.
    CHECK(spans[4].begin == doctest::Approx(0.22f));
    CHECK(spans[4].end == doctest::Approx(0.68f));
    // Neighbours are untouched and still index-aligned with their tokens.
    CHECK(spans[3].end == doctest::Approx(0.22f));
    CHECK(spans[5].begin == doctest::Approx(0.68f));
}

TEST_CASE("injectGapSpaces ignores a spans vector of the wrong length") {
    std::vector<std::string> tokens = {"A", "B", "C", "D", "1", "2", "3", "4"};
    std::vector<float> pos = {0.05f, 0.10f, 0.15f, 0.20f, 0.70f, 0.75f, 0.80f, 0.85f};
    std::vector<TokenSpan> spans(3); // not index-aligned
    injectGapSpaces(tokens, pos, nullptr, &spans);
    // Same guard as `scores`: bail out entirely rather than desync the vectors.
    CHECK(tokens.size() == 8);
    CHECK(pos.size() == 8);
    CHECK(spans.size() == 3);
}
