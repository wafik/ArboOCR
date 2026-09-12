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

    /// `intraOpNumThreads`/`interOpNumThreads` size the ORT thread pools;
    /// 0 (the default) leaves the choice to ORT. See EngineConfig for when
    /// setting them is worth it. Negative values are clamped to 0.
    /// `enableCpuMemArena` false (the default) keeps DisableCpuMemArena for
    /// bounded RSS; see EngineConfig::enableCpuMemArena.
    void loadModel(const std::string& modelPath, bool useCuda = false,
                   bool useTensorrt = false, const std::string& trtCacheDir = "",
                   bool useFp16 = true, int intraOpNumThreads = 0,
                   int interOpNumThreads = 0, bool enableCpuMemArena = false);

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

    /// Max crops per recognition inference call (default 6, matching
    /// PaddleOCR/RapidOCR). Call before loadModel() so TensorRT profiles
    /// are built against the same batch size used at runtime. Values < 1
    /// are clamped to 1.
    void setRecBatchNum(int n);
    int recBatchNum() const { return recBatchNum_; }

    /// Opt-in per-token output: when enabled, every RawTextLine also carries
    /// `tokens` (one entry per emitted CTC token) and `spans` (that token's
    /// horizontal extent as a fraction of the crop's *content* width), so a
    /// caller can map tokens back to page coordinates for word boxes.
    /// Off by default: deriving the spans during the CTC decode costs
    /// essentially nothing (it's bookkeeping the decode loop already has the
    /// information for), but carrying two extra vectors per line for every
    /// line of every page is not free — so it's paid for only when the
    /// caller actually wants word boxes.
    void setReturnSpans(bool enabled);
    bool returnSpans() const { return returnSpans_; }

    /// Opt-in inter-word space recovery. Recognition models routinely drop the
    /// spaces between words: at a word boundary the space class scores just
    /// under the following letter, so greedy CTC argmax swallows it and
    /// "ATAS NAMA" comes back as "ATASNAMA". When enabled, a timestep whose
    /// winner is a real character but whose *space* logit is a strong
    /// runner-up (above ppu-paddle-ocr's 0.001 bar, whose spaceRecovery this
    /// mirrors) emits that space alongside the character.
    /// Off by default and byte-identical when off: the decode output is
    /// compared against fixtures, and the heuristic can insert spurious
    /// spaces in dense symbol/number runs where the space class is a common
    /// near-miss. Turn it on when output words are visibly run together.
    void setSpaceRecovery(bool enabled);
    bool spaceRecovery() const { return spaceRecovery_; }

    /// Recognize one cropped, angle-corrected text-line image per input Mat,
    /// returned in the SAME order as `partImages`.
    ///
    /// Internally batches up to recBatchNum_ crops per ONNXRuntime
    /// Session::Run() call instead of one call per crop — mirrors
    /// PaddleOCR/RapidOCR's TextRecognizer.__call__ / resize_norm_img
    /// (tools/infer/predict_rec.py and python/rapidocr/ch_ppocr_rec/main.py
    /// upstream) exactly:
    ///   1. Sort crops by ascending aspect ratio (width/height) so each
    ///      batch groups similarly-shaped crops together (less padding
    ///      waste); original order is restored in the returned vector.
    ///   2. Per batch, `max_wh_ratio` starts at `kMinBatchWidth / kDstHeight`
    ///      and is raised to the widest crop's own ratio in that batch;
    ///      `batchWidth = int(kDstHeight * max_wh_ratio)` (truncating cast,
    ///      not rounding), clamped up to the widest crop's own resized width.
    ///      The seed is a 32px floor rather than the model's reference ratio
    ///      (320/48): a narrow batch is sized to the widest crop it actually
    ///      contains (ppu batched.ts:55-59) instead of always paying for a
    ///      320px-wide strip of padding.
    ///   3. Each crop is resized (aspect-ratio preserved, height fixed) to
    ///      its own natural width capped at `batchWidth`, THEN normalized —
    ///      only after that is it copied into a zero-initialized
    ///      `batchWidth`-wide buffer. Padding value is 0.0 in NORMALIZED
    ///      float space, not a pre-normalization pixel value — those are
    ///      different values (raw pixel 0 normalizes to -1.0, not 0.0).
    ///      `batchWidth` tracks the batch's own content, so a crop's own
    ///      resized pixels are the same whatever the batch's width is.
    ///   4. CTC-decoded per row only up to that row's share of the padded
    ///      width (`ceil(timeSteps * crop.cols / batchWidth)`). The trailing
    ///      timesteps cover only zero padding, so cutting them before the
    ///      decode removes argmax work — ppu truncates the same way
    ///      (batched.ts:99-107).
    ///
    /// NOT byte-identical to the old fixed-320 strip, and the unit tests
    /// cannot tell you so: they feed synthetic fixtures through
    /// decodeForTest(), never real model logits, so they pin the decode
    /// algebra (padding must not move content) but not the text a real CRNN
    /// emits. Measured on SROIE small (40 receipts, out/bench_arbo_ppu_ab.py
    /// vs out/bench_ppu_n40_small3.json, oar/ppu re-run as controls to rule
    /// out thermal drift): avg sim 86.31% -> 86.40% — flat to slightly up —
    /// but only 8/40 stems identical, 20 up and 12 down, worst -0.5pp.
    /// Cause is the batch-width change, not the truncation: a narrower strip
    /// feeds the conv stack different right-edge context, so logits differ
    /// marginally in the last characters of each crop. Net neutral here;
    /// re-measure on your own corpus before relying on either number.
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
    /// model. `contentFraction` defaults to 1.0 (crop exactly fills the
    /// batch strip, no padding to divide out) so existing call sites are
    /// unaffected; pass < 1.0 to exercise the padding correction in
    /// scoreToTextLine().
    RawTextLine decodeForTest(const std::vector<float>& outputData, size_t h, size_t w,
                              float contentFraction = 1.0f) const {
        return scoreToTextLine(outputData.data(), outputData.size(), h, w, contentFraction);
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
    ///
    /// `h` is the CTC timestep count of the PADDED batch strip — runBatchInference()
    /// passes the model's full `timeSteps`, not this crop's share — and this
    /// function cuts its own decode at `ceil(h * contentFraction)`. Do NOT
    /// pre-truncate `h` at the call site: it is also the denominator of every
    /// timestep fraction below (see the `invH`/span math), so shrinking it
    /// would rescale the spans and silently move every word box.
    /// `contentFraction` is `crop.cols / batchWidth` — the fraction of the
    /// padded strip actually occupied by image content — and drives both that
    /// decode cut and the span rescale onto the crop's own content width, so
    /// the two can never disagree. A single precomputed ratio
    /// is passed rather than the two widths because that is the only thing the
    /// decode needs: it keeps this function ignorant of batching entirely, and
    /// it gives the test seam one knob instead of two coupled ones. Values <= 0
    /// are treated as 1.0 (no correction, decode everything).
    RawTextLine scoreToTextLine(const float* outputData, size_t dataSize, size_t h, size_t w,
                                float contentFraction = 1.0f) const;

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

    // Default matches PaddleOCR/RapidOCR's rec_batch_num=6. Overridable via
    // setRecBatchNum() / EngineConfig::recBatchNum before loadModel().
    int recBatchNum_ = 6;
    // Off by default — see setReturnSpans().
    bool returnSpans_ = false;
    // Off by default — see setSpaceRecovery().
    bool spaceRecovery_ = false;
    // Smallest batch strip width the recognizer will run at: a batch is sized
    // to the widest crop it actually holds (ppu batched.ts:55-59), with this
    // narrow floor so a batch of tiny crops can't collapse to a degenerate
    // strip. Deliberately NOT the model's reference width (rec_image_shape's
    // 320) — flooring every batch there padded narrow batches out to 320px of
    // pure wasted inference. The single min/max profile registered with
    // TensorRT (`x:1x3x48x32`) uses the same value.
    static constexpr int kMinBatchWidth = 32;
    static constexpr int kDstHeight = 48;
    const float meanValues_[3] = {127.5f, 127.5f, 127.5f};
    const float normValues_[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
};

} // namespace arbo::ocr
