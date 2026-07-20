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
