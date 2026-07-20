#pragma once
// Adapted from RapidAI/RapidOcrOnnx's CrnnNet.h/.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. Renamed to Recognizer for arboOCR's public API.
// scoreToTextLine/getTextLine are near-verbatim ports. loadKeysFromModelMetadata()
// is new: RapidOcrOnnx only supports loadKeysFromFile() (a separate
// keys.txt); PP-OCRv6 models embed the dict in ONNX metadata instead.

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "arboOCR/types.hpp"

namespace arbo::ocr {

class Recognizer {
public:
    Recognizer();
    ~Recognizer();

    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    void loadModel(const std::string& modelPath, bool useCuda = false,
                   bool useTensorrt = false, const std::string& trtCacheDir = "");

    /// RapidOcrOnnx's original keys-loading mechanism: one character per
    /// line in a plain text file. On success, prepends "#" (CTC blank) and
    /// appends " " (space), matching RapidOcrOnnx's exact layout. Missing
    /// file leaves keys empty (does not throw) so callers can detect
    /// failure via `keyCount() == 0` and try another source.
    void loadKeysFromFile(const std::string& keysPath);

    /// Reads the `character` field from the currently-loaded ONNX model's
    /// custom_metadata_map (this is how PP-OCRv6/rapidocr-v3 models embed
    /// their dict). Must be called after loadModel(). Applies the same
    /// "#" prefix + " " suffix convention as loadKeysFromFile() so the CTC
    /// decode logic is identical regardless of dict source. Returns false
    /// (leaving keys empty) if the model has no such metadata — callers
    /// should then fall back to loadKeysFromFile() with a bundled
    /// PP-OCRv4/v5-style keys.txt.
    bool loadKeysFromModelMetadata();

    /// Number of loaded keys (0 if neither load method has succeeded yet).
    size_t keyCount() const { return keys_.size(); }

    /// Recognize one cropped, angle-corrected text-line image per input Mat,
    /// returned in the SAME order as `partImages`.
    ///
    /// Internally batches up to kRecBatchNum crops per ONNXRuntime
    /// Session::Run() call instead of one call per crop — mirrors
    /// PaddleOCR/RapidOCR's TextRecognizer.__call__ / resize_norm_img
    /// (tools/infer/predict_rec.py and python/rapidocr/ch_ppocr_rec/main.py
    /// upstream) exactly:
    ///   1. Sort crops by ascending aspect ratio (width/height) so each
    ///      batch groups similarly-shaped crops together (less padding
    ///      waste); original order is restored in the returned vector.
    ///   2. Per batch, `max_wh_ratio` starts at the model's reference ratio
    ///      (320/48) and is raised to the widest crop's own ratio in that
    ///      batch; `batchWidth = int(kDstHeight * max_wh_ratio)` (truncating
    ///      cast, not rounding).
    ///   3. Each crop is resized (aspect-ratio preserved, height fixed) to
    ///      its own natural width capped at `batchWidth`, THEN normalized —
    ///      only after that is it copied into a zero-initialized
    ///      `batchWidth`-wide buffer. Padding value is 0.0 in NORMALIZED
    ///      float space, not a pre-normalization pixel value — those are
    ///      different values (raw pixel 0 normalizes to -1.0, not 0.0).
    ///   4. CTC-decoded per row over the full padded width with no
    ///      truncation/masking by real width — this relies on the trained
    ///      CRNN model reliably emitting the CTC blank token in padding
    ///      columns, which is how upstream does it too (confirmed by
    ///      absence of any masking code there, not by an explicit guarantee
    ///      in the decode step itself).
    ///
    /// Measured on the bundled sample receipt (31 text lines) on a Jetson
    /// Nano: CPU backend got SLOWER after batching (~3.9s -> ~5.0s) — CPU
    /// has no real parallelism across the batch dimension, so padding every
    /// crop up to its batch's shared width is pure wasted computation there.
    /// TensorRT backend got faster (~456ms -> ~340-400ms across runs) —
    /// batching only pays off where the backend can actually parallelize
    /// across the batch dimension (GPU/TensorRT), not on CPU. If you're
    /// running Engine on CPU only, this trade-off is baked in; there's no
    /// config flag to disable batching currently.
    std::vector<RawTextLine> getTextLines(std::vector<cv::Mat>& partImages);

    /// Test-only entry point: run the CTC decode directly on a raw output
    /// buffer without going through ONNXRuntime inference. Exposed so
    /// test_recognizer.cpp can validate decode correctness without a real
    /// model.
    RawTextLine decodeForTest(const std::vector<float>& outputData, size_t h, size_t w) const {
        return scoreToTextLine(outputData.data(), outputData.size(), h, w);
    }

    /// Test-only entry point: build the padded, normalized batch tensor
    /// (the buffer that would be fed to ONNXRuntime) without touching a
    /// session. Exposed so test_recognizer.cpp can catch buffer-layout bugs
    /// (e.g. a row-copy misalignment when a crop's width < batchWidth) with
    /// only synthetic images, no real ONNX model required — this exact kind
    /// of bug was originally caught only by real-model inference on
    /// hardware, which is too slow/expensive to be the sole safety net.
    std::vector<float> buildBatchTensorForTest(const std::vector<cv::Mat>& resizedCrops, int batchWidth) const {
        return buildBatchTensor(resizedCrops, batchWidth);
    }

private:
    /// `outputData`/`dataSize` is a raw pointer+size rather than a
    /// std::vector& so callers slicing one batch item's rows out of a
    /// larger ONNXRuntime output buffer (see runBatchInference()) don't
    /// need to materialize a per-item copy just to call this.
    RawTextLine scoreToTextLine(const float* outputData, size_t dataSize, size_t h, size_t w) const;

    /// Builds the padded, normalized [batchSize, 3, kDstHeight, batchWidth]
    /// CHW tensor buffer from already-resized crops: each crop is
    /// normalized first, then copied row-by-row into a zero-initialized
    /// batchWidth-wide slot (padding = 0.0 in NORMALIZED float space — see
    /// getTextLines() doc comment for why this differs from padding with a
    /// pre-normalization pixel value). Pure/stateless — no ONNXRuntime
    /// session involved, which is what makes it independently testable.
    ///
    /// Precondition per crop: exactly kDstHeight rows, and cols <=
    /// batchWidth. getTextLines() (the only real caller) always resizes to
    /// kDstHeight and caps width at batchWidth before calling this, so the
    /// precondition holds there — but this method is also reachable
    /// directly via the public buildBatchTensorForTest() test seam, so a
    /// violating crop is skipped (left as zero-padding) rather than trusted:
    /// getting this wrong would otherwise read past the end of
    /// substractMeanNormalize()'s crop.cols*crop.rows-sized output buffer.
    std::vector<float> buildBatchTensor(const std::vector<cv::Mat>& resizedCrops, int batchWidth) const;

    /// Builds the batch tensor, runs ONE Session::Run() call, and CTC-decodes
    /// every row. `resizedCrops[i].cols` may be < batchWidth (narrower crops
    /// get right-padded with zeros in NORMALIZED float space — see
    /// getTextLines() doc comment). Throws Ort::Exception if the loaded
    /// model doesn't accept the given batch size (e.g. a custom export with
    /// a fixed batch=1 input dim) — callers should catch and fall back to
    /// single-image calls; see runBatch().
    std::vector<RawTextLine> runBatchInference(const std::vector<cv::Mat>& resizedCrops, int batchWidth);

    /// Wraps runBatchInference() with a fallback: if the model rejects the
    /// batch (Ort::Exception — happens only for non-standard custom models;
    /// every official PP-OCR model supports dynamic batch, confirmed by
    /// this project's own TensorRT profile documentation), retries one crop
    /// at a time instead of losing the whole batch's results.
    std::vector<RawTextLine> runBatch(const std::vector<cv::Mat>& resizedCrops, int batchWidth);

    void finalizeKeys(std::vector<std::string> rawKeys);

    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR, "Recognizer"};
    Ort::SessionOptions sessionOptions_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;
    std::vector<std::string> keys_;

    // Mirrors PaddleOCR/RapidOCR's rec_batch_num=6 default (rec_image_shape
    // [3, 48, 320], i.e. imgW=320 -> the reference aspect ratio used to seed
    // max_wh_ratio for each batch — see getTextLines() doc comment).
    static constexpr int kRecBatchNum = 6;
    static constexpr int kRefImgWidth = 320;
    static constexpr int kDstHeight = 48;
    const float meanValues_[3] = {127.5f, 127.5f, 127.5f};
    const float normValues_[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
};

} // namespace arbo::ocr
