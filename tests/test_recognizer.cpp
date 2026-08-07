// tests/test_recognizer.cpp
#include <doctest/doctest.h>
#include "arboOCR/recognizer.hpp"

using namespace arbo::ocr;

TEST_CASE("loadKeysFromFile loads keys and prepends blank + appends space") {
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    CHECK(net.keyCount() == 7); // 5 chars + blank prefix + space suffix
}

TEST_CASE("loadKeysFromFile on a missing file leaves keys empty, does not throw") {
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/does_not_exist.txt");
    CHECK(net.keyCount() == 0);
}

TEST_CASE("loadKeysFromModelMetadata without a loaded model returns false") {
    Recognizer net; // never loadModel()
    CHECK(net.loadKeysFromModelMetadata() == false);
    CHECK(net.keyCount() == 0);
}

TEST_CASE("CTC greedy decode collapses repeats and skips blank index 0") {
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    // keys = ["#", "a", "b", "c", "d", "e", " "]  (indices 0..6)
    std::vector<float> output(6 * 7, 0.0f);
    auto setArgmax = [&](int t, int idx) { output[t * 7 + idx] = 1.0f; };
    setArgmax(0, 1); // a
    setArgmax(1, 1); // a (repeat, collapsed)
    setArgmax(2, 2); // b
    setArgmax(3, 0); // blank
    setArgmax(4, 3); // c
    setArgmax(5, 3); // c (repeat, collapsed)
    auto line = net.decodeForTest(output, 6, 7);
    CHECK(line.text == "abc");
}

// --- Batch tensor buffer-layout tests -----------------------------------
// These exercise buildBatchTensorForTest() with synthetic crops, no ONNX
// model needed. They exist specifically because the original batching
// implementation had a row-copy misalignment bug (copying each crop's
// normalized plane as one contiguous block instead of row-by-row) that
// silently corrupted every row after the first whenever a crop's resized
// width was less than the batch's shared width — real-model inference on
// hardware caught it (garbled recognized text), but that shouldn't be the
// only safety net for a memory-layout bug like this.

namespace {
// meanValues_/normValues_ are {127.5,127.5,127.5} / {1/127.5,...}, so a
// uniform pixel value of `v` normalizes to (v - 127.5) / 127.5. Using 255
// (-> 1.0) and 0 (-> -1.0) gives easily distinguishable, exact expected
// values for asserting buffer contents without needing to know the exact
// normalization constants inside the test.
constexpr float kNormalizedWhite = 1.0f;  // pixel 255
constexpr float kNormalizedBlack = -1.0f; // pixel 0

cv::Mat solidCrop(int width, int height, uchar value) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(value, value, value));
}
} // namespace

TEST_CASE("buildBatchTensor: single crop exactly filling batchWidth has no padding") {
    Recognizer net;
    int batchWidth = 32;
    auto crop = solidCrop(batchWidth, 48, 255);
    auto buffer = net.buildBatchTensorForTest({crop}, batchWidth);
    REQUIRE(buffer.size() == static_cast<size_t>(1 * 3 * 48 * batchWidth));
    for (float v : buffer) {
        CHECK(v == doctest::Approx(kNormalizedWhite));
    }
}

TEST_CASE("buildBatchTensor: narrower crop is left-aligned, padding is 0.0 not black-pixel-normalized") {
    Recognizer net;
    int batchWidth = 64;
    int cropWidth = 32; // half of batchWidth
    auto crop = solidCrop(cropWidth, 48, 255); // all-white crop, narrower than the batch
    auto buffer = net.buildBatchTensorForTest({crop}, batchWidth);
    REQUIRE(buffer.size() == static_cast<size_t>(1 * 3 * 48 * batchWidth));

    // Per channel, per row: first cropWidth columns should be the crop's
    // normalized value (white -> 1.0); the remaining columns are padding
    // and must be exactly 0.0 (NOT kNormalizedBlack, which is what a
    // black *pixel* would normalize to — the padding convention is 0.0 in
    // already-normalized float space, a distinct value).
    for (int ch = 0; ch < 3; ch++) {
        for (int row = 0; row < 48; row++) {
            const float* rowPtr = buffer.data() + (static_cast<size_t>(ch) * 48 + row) * batchWidth;
            for (int col = 0; col < cropWidth; col++) {
                CHECK(rowPtr[col] == doctest::Approx(kNormalizedWhite));
            }
            for (int col = cropWidth; col < batchWidth; col++) {
                CHECK(rowPtr[col] == doctest::Approx(0.0f));
                CHECK(rowPtr[col] != doctest::Approx(kNormalizedBlack));
            }
        }
    }
}

TEST_CASE("buildBatchTensor: each row lands in the correct row of the padded buffer (regression guard)") {
    // This is the exact scenario that exposed the original row-copy bug:
    // crop.cols != batchWidth means the source (packed) and destination
    // (padded) planes have different row strides. A single contiguous
    // copy per channel would shift every row after row 0 into the wrong
    // place in the destination. Build a crop where each row has a distinct
    // pixel value, then confirm every row of every channel lands at its
    // own row offset in the batch buffer, not some shifted position.
    Recognizer net;
    int batchWidth = 40;
    int cropWidth = 20; // < batchWidth, so padding (and the bug, if present) is exercised
    int height = 48;

    cv::Mat crop(height, cropWidth, CV_8UC3);
    for (int row = 0; row < height; row++) {
        // Distinct, deterministic value per row: row 0 -> 0, row 47 -> ~254.
        uchar value = static_cast<uchar>((row * 255) / (height - 1));
        crop.row(row).setTo(cv::Scalar(value, value, value));
    }

    auto buffer = net.buildBatchTensorForTest({crop}, batchWidth);
    REQUIRE(buffer.size() == static_cast<size_t>(1 * 3 * height * batchWidth));

    for (int ch = 0; ch < 3; ch++) {
        for (int row = 0; row < height; row++) {
            uchar expectedPixel = static_cast<uchar>((row * 255) / (height - 1));
            float expectedNormalized = (static_cast<float>(expectedPixel) - 127.5f) / 127.5f;
            const float* rowPtr = buffer.data() + (static_cast<size_t>(ch) * height + row) * batchWidth;
            // Every column within the real crop width, for THIS row, must
            // match this row's expected value — if the bug were present,
            // columns here would instead hold a neighboring row's value.
            for (int col = 0; col < cropWidth; col++) {
                CHECK(rowPtr[col] == doctest::Approx(expectedNormalized));
            }
            // Padding columns for this row must be exactly 0.0.
            for (int col = cropWidth; col < batchWidth; col++) {
                CHECK(rowPtr[col] == doctest::Approx(0.0f));
            }
        }
    }
}

TEST_CASE("buildBatchTensor: multiple crops of different widths occupy independent, correctly-strided slots") {
    Recognizer net;
    int batchWidth = 48;
    auto cropA = solidCrop(16, 48, 255); // white, narrow
    auto cropB = solidCrop(48, 48, 0);   // black, full width (no padding)
    auto buffer = net.buildBatchTensorForTest({cropA, cropB}, batchWidth);
    REQUIRE(buffer.size() == static_cast<size_t>(2 * 3 * 48 * batchWidth));

    size_t cropStride = static_cast<size_t>(3) * 48 * batchWidth;
    // Crop A (batch index 0): white in its real columns, 0.0 padding after.
    for (int ch = 0; ch < 3; ch++) {
        for (int row = 0; row < 48; row++) {
            const float* rowPtr = buffer.data() + 0 * cropStride + (static_cast<size_t>(ch) * 48 + row) * batchWidth;
            for (int col = 0; col < 16; col++) CHECK(rowPtr[col] == doctest::Approx(kNormalizedWhite));
            for (int col = 16; col < batchWidth; col++) CHECK(rowPtr[col] == doctest::Approx(0.0f));
        }
    }
    // Crop B (batch index 1): black across the full width, no padding.
    for (int ch = 0; ch < 3; ch++) {
        for (int row = 0; row < 48; row++) {
            const float* rowPtr = buffer.data() + 1 * cropStride + (static_cast<size_t>(ch) * 48 + row) * batchWidth;
            for (int col = 0; col < batchWidth; col++) CHECK(rowPtr[col] == doctest::Approx(kNormalizedBlack));
        }
    }
}

TEST_CASE("buildBatchTensor: a crop with the wrong height is skipped, not an out-of-bounds read") {
    // buildBatchTensor's row-copy math assumes every crop has exactly
    // kDstHeight (48) rows — getTextLines() guarantees this via its resize
    // step, but buildBatchTensorForTest() is a public test seam that can be
    // called directly with anything. A crop with the wrong height used to
    // have no guard here at all: substractMeanNormalize()'s output is sized
    // crop.cols*crop.rows, but the row-copy loop indexed it assuming
    // crop.rows==kDstHeight — a mismatch would silently read past the end
    // of that buffer. This asserts the guard skips such a crop cleanly
    // (leaving its slot as zero-padding) instead of crashing/corrupting.
    Recognizer net;
    int batchWidth = 32;
    cv::Mat wrongHeightCrop = solidCrop(32, 24, 255); // 24 rows, not 48
    std::vector<float> buffer;
    CHECK_NOTHROW(buffer = net.buildBatchTensorForTest({wrongHeightCrop}, batchWidth));
    REQUIRE(buffer.size() == static_cast<size_t>(1 * 3 * 48 * batchWidth));
    // Skipped crop's entire slot should be untouched zero-padding.
    for (float v : buffer) {
        CHECK(v == doctest::Approx(0.0f));
    }
}

TEST_CASE("buildBatchTensor: a crop wider than batchWidth is skipped, not an out-of-bounds write") {
    Recognizer net;
    int batchWidth = 16;
    cv::Mat tooWideCrop = solidCrop(32, 48, 255); // wider than batchWidth
    std::vector<float> buffer;
    CHECK_NOTHROW(buffer = net.buildBatchTensorForTest({tooWideCrop}, batchWidth));
    REQUIRE(buffer.size() == static_cast<size_t>(1 * 3 * 48 * batchWidth));
    for (float v : buffer) {
        CHECK(v == doctest::Approx(0.0f));
    }
}

// --- Per-token span tests (setReturnSpans / RawTextLine::spans) -----------
// These pin down the two things that are easy to get silently wrong when
// mapping decoded tokens back to page coordinates:
//   1. A token's span must cover its whole CTC run, not just the single
//      timestep the decode loop emits on.
//   2. Timesteps span the PADDED batch width, so a crop narrower than its
//      batch's shared width has every fraction compressed toward 0 unless
//      the batchWidth/crop.cols correction is applied.
// Everything below drives decodeForTest(), so no ONNX model is involved.

namespace {
// Build a synthetic CTC logit buffer laid out [timesteps, numClasses]:
// `argmaxPerStep[t]` is the class index that wins at timestep t (0 = CTC
// blank). It gets 1.0 and every other class stays at 0.0, so the argmax is
// unambiguous and the recorded score is exactly 1.0.
std::vector<float> ctcOutput(const std::vector<int>& argmaxPerStep, int numClasses) {
    std::vector<float> out(argmaxPerStep.size() * static_cast<size_t>(numClasses), 0.0f);
    for (size_t t = 0; t < argmaxPerStep.size(); t++) {
        out[t * static_cast<size_t>(numClasses) + static_cast<size_t>(argmaxPerStep[t])] = 1.0f;
    }
    return out;
}
} // namespace

TEST_CASE("returnSpans is off by default: tokens/spans stay empty and text is unchanged") {
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    // keys = ["#", "a", "b", "c", "d", "e", " "]  (indices 0..6)
    CHECK(net.returnSpans() == false);
    // 6 timesteps: 'a' owns 0-1, blank at 2, 'b' owns 3-5.
    auto output = ctcOutput({1, 1, 0, 2, 2, 2}, 7);
    auto line = net.decodeForTest(output, 6, 7);
    CHECK(line.text == "ab");
    CHECK(line.charScores.size() == 2);
    // The opt-in guard: no caller asked for word boxes, so nothing extra is
    // carried on the result.
    CHECK(line.tokens.empty());
    CHECK(line.spans.empty());
}

TEST_CASE("returnSpans on, no padding: each span covers its token's whole CTC run") {
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    net.setReturnSpans(true);
    CHECK(net.returnSpans() == true);
    // Same logits as the off-by-default case above: 'a' owns timesteps 0-1,
    // 'b' owns timesteps 3-5. contentFraction defaults to 1.0 (the crop
    // exactly filled the batch strip, C == W), so no padding correction.
    auto output = ctcOutput({1, 1, 0, 2, 2, 2}, 7);
    auto line = net.decodeForTest(output, 6, 7);

    CHECK(line.text == "ab"); // identical to the spans-off run: text never changes
    REQUIRE(line.tokens.size() == 2);
    REQUIRE(line.spans.size() == 2);
    CHECK(line.charScores.size() == 2);
    CHECK(line.tokens[0] == "a");
    CHECK(line.tokens[1] == "b");

    // 'a' occupies timesteps [0,2) of 6 -> [0, 1/3]. 'b' occupies [3,6) of 6
    // -> [1/2, 1]; its run has no following timestep to close it, so it must
    // close at the end of the sequence rather than one timestep after it
    // started.
    CHECK(line.spans[0].begin == doctest::Approx(0.0f));
    CHECK(line.spans[0].end == doctest::Approx(2.0f / 6.0f));
    CHECK(line.spans[1].begin == doctest::Approx(3.0f / 6.0f));
    CHECK(line.spans[1].end == doctest::Approx(1.0f));

    // Ordered, non-overlapping, inside [0,1].
    CHECK(line.spans[0].begin < line.spans[0].end);
    CHECK(line.spans[1].begin < line.spans[1].end);
    CHECK(line.spans[0].end <= line.spans[1].begin);
    for (const auto& s : line.spans) {
        CHECK(s.begin >= 0.0f);
        CHECK(s.end <= 1.0f);
    }

    // Widths proportional to run lengths: 2 timesteps vs 3.
    const float widthA = line.spans[0].end - line.spans[0].begin;
    const float widthB = line.spans[1].end - line.spans[1].begin;
    CHECK(widthA == doctest::Approx(2.0f / 6.0f));
    CHECK(widthB == doctest::Approx(3.0f / 6.0f));
    CHECK(widthB / widthA == doctest::Approx(1.5f));
}

TEST_CASE("returnSpans with padding: spans are rescaled onto the crop's content width") {
    // Scenario: the crop was resized to C columns and zero-padded out to
    // W = 2*C before inference, so the model's timesteps cover the PADDED
    // strip — the glyphs sit in the first half and the tail decodes to CTC
    // blank (what a trained CRNN emits over padding columns). Without the
    // W/C correction every span lands at half its true content-relative
    // position. This is the case that catches that bug.
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    net.setReturnSpans(true);
    // 12 timesteps: 'a' owns 0-1, 'b' owns 3-5, timesteps 6-11 are padding.
    auto output = ctcOutput({1, 1, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0}, 7);

    // Uncorrected (contentFraction = 1.0) -> fractions of the PADDED strip.
    auto padded = net.decodeForTest(output, 12, 7, 1.0f);
    REQUIRE(padded.spans.size() == 2);
    CHECK(padded.spans[0].begin == doctest::Approx(0.0f));
    CHECK(padded.spans[0].end == doctest::Approx(2.0f / 12.0f));
    CHECK(padded.spans[1].begin == doctest::Approx(3.0f / 12.0f));
    CHECK(padded.spans[1].end == doctest::Approx(6.0f / 12.0f));

    // Corrected (contentFraction = C/W = 0.5) -> exactly double every fraction.
    auto corrected = net.decodeForTest(output, 12, 7, 0.5f);
    REQUIRE(corrected.spans.size() == 2);
    for (size_t i = 0; i < corrected.spans.size(); i++) {
        CHECK(corrected.spans[i].begin == doctest::Approx(2.0f * padded.spans[i].begin));
        CHECK(corrected.spans[i].end == doctest::Approx(2.0f * padded.spans[i].end));
        CHECK(corrected.spans[i].begin >= 0.0f);
        CHECK(corrected.spans[i].end <= 1.0f);
    }
    CHECK(corrected.spans[0].begin == doctest::Approx(0.0f));
    CHECK(corrected.spans[0].end == doctest::Approx(1.0f / 3.0f));
    CHECK(corrected.spans[1].begin == doctest::Approx(0.5f));
    CHECK(corrected.spans[1].end == doctest::Approx(1.0f));

    // The strongest form of the same statement: the corrected spans must
    // equal what the identical glyph layout produces with no padding at all
    // (6 timesteps, C == W). Padding a crop must not move its content.
    auto unpadded = net.decodeForTest(ctcOutput({1, 1, 0, 2, 2, 2}, 7), 6, 7, 1.0f);
    REQUIRE(unpadded.spans.size() == 2);
    for (size_t i = 0; i < unpadded.spans.size(); i++) {
        CHECK(corrected.spans[i].begin == doctest::Approx(unpadded.spans[i].begin));
        CHECK(corrected.spans[i].end == doctest::Approx(unpadded.spans[i].end));
    }

    // The correction touches spans only; decoded text is unaffected.
    CHECK(padded.text == "ab");
    CHECK(corrected.text == "ab");
}

TEST_CASE("returnSpans: a repeat separated by a blank yields two distinct, ordered spans") {
    // `a a # a a #` — the blank at timestep 2 breaks the CTC run, so this
    // decodes to two separate 'a' tokens. Their spans must be disjoint and
    // ordered, not one merged extent covering both: the run-tracking has to
    // close a run when the argmax changes to blank, not only when it changes
    // to another character.
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    net.setReturnSpans(true);
    auto output = ctcOutput({1, 1, 0, 1, 1, 0}, 7);
    auto line = net.decodeForTest(output, 6, 7);

    CHECK(line.text == "aa");
    REQUIRE(line.tokens.size() == 2);
    REQUIRE(line.spans.size() == 2);
    CHECK(line.tokens[0] == "a");
    CHECK(line.tokens[1] == "a");

    CHECK(line.spans[0].begin == doctest::Approx(0.0f));
    CHECK(line.spans[0].end == doctest::Approx(2.0f / 6.0f));
    CHECK(line.spans[1].begin == doctest::Approx(3.0f / 6.0f));
    CHECK(line.spans[1].end == doctest::Approx(5.0f / 6.0f));
    // Strictly separated by the blank timestep between the two runs.
    CHECK(line.spans[0].end < line.spans[1].begin);
    CHECK(line.spans[0].begin != line.spans[1].begin);
}

TEST_CASE("returnSpans: multi-byte CJK tokens stay whole codepoints and index-aligned") {
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/cjk_keys.txt");
    REQUIRE(net.keyCount() == 5); // 3 CJK chars + blank prefix + space suffix
    net.setReturnSpans(true);
    // keys = ["#", U+4E2D, U+6587, U+5B57, " "]  (indices 0..4)
    // 6 timesteps: U+4E2D owns 0-1, blank, U+6587 at 3, blank, U+5B57 at 5.
    auto output = ctcOutput({1, 1, 0, 2, 0, 3}, 5);
    auto line = net.decodeForTest(output, 6, 5);

    // Byte escapes rather than raw literals, so these assertions don't depend
    // on how a given compiler interprets this source file's encoding.
    const std::string kZhong = "\xE4\xB8\xAD"; // U+4E2D
    const std::string kWen = "\xE6\x96\x87";   // U+6587
    const std::string kZi = "\xE5\xAD\x97";    // U+5B57

    REQUIRE(line.tokens.size() == 3);
    CHECK(line.spans.size() == line.tokens.size());
    CHECK(line.charScores.size() == line.tokens.size());

    // Whole UTF-8 codepoints per token, never split into individual bytes.
    CHECK(line.tokens[0] == kZhong);
    CHECK(line.tokens[1] == kWen);
    CHECK(line.tokens[2] == kZi);
    for (const auto& t : line.tokens) {
        CHECK(t.size() == 3);
    }
    CHECK(line.text == kZhong + kWen + kZi);

    CHECK(line.spans[0].begin == doctest::Approx(0.0f));
    CHECK(line.spans[0].end == doctest::Approx(2.0f / 6.0f));
    CHECK(line.spans[1].begin == doctest::Approx(3.0f / 6.0f));
    CHECK(line.spans[1].end == doctest::Approx(4.0f / 6.0f));
    CHECK(line.spans[2].begin == doctest::Approx(5.0f / 6.0f));
    CHECK(line.spans[2].end == doctest::Approx(1.0f)); // last run closes at the final timestep
    for (size_t i = 1; i < line.spans.size(); i++) {
        CHECK(line.spans[i - 1].end <= line.spans[i].begin);
    }
}

TEST_CASE("returnSpans: injected gap spaces stay index-aligned with tokens and scores") {
    // >= 4 tokens with one wide gap, so injectGapSpaces actually fires and
    // has to splice a span in alongside the token and the score. Anything
    // that inserts into two of the three vectors but not the third shows up
    // here as a size mismatch, which would misalign every box after the gap.
    Recognizer net;
    net.loadKeysFromFile("tests/fixtures/sample_keys.txt");
    net.setReturnSpans(true);
    // 20 timesteps: a@0, b@2, c@4, a wide blank valley, then d@12, e@14.
    auto output = ctcOutput({1, 0, 2, 0, 3, 0, 0, 0, 0, 0,
                             0, 0, 4, 0, 5, 0, 0, 0, 0, 0}, 7);
    auto line = net.decodeForTest(output, 20, 7);

    CHECK(line.text == "abc de");
    REQUIRE(line.tokens.size() == 6); // 5 decoded + 1 injected space
    CHECK(line.spans.size() == line.tokens.size());
    CHECK(line.charScores.size() == line.tokens.size());
    CHECK(line.tokens[3] == " "); // the injected token, at the gap

    // Still ordered and inside [0,1] across the injection point.
    for (size_t i = 0; i < line.spans.size(); i++) {
        CHECK(line.spans[i].begin >= 0.0f);
        CHECK(line.spans[i].end <= 1.0f);
        CHECK(line.spans[i].begin <= line.spans[i].end);
        if (i > 0) {
            CHECK(line.spans[i - 1].begin <= line.spans[i].begin);
        }
    }
}
