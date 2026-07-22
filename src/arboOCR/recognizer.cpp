// Adapted from RapidAI/RapidOcrOnnx's CrnnNet.cpp (Apache-2.0) — see
// THIRD_PARTY_NOTICES.md. Renamed to Recognizer for arboOCR's public API.
// loadKeysFromModelMetadata() is new code, not present in the upstream
// project (see recognizer.hpp doc comment).
#include "arboOCR/recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "arboOCR/ocr_utils.hpp"
#include "ort_provider_utils.hpp"

namespace arbo::ocr {

namespace {

template <typename ForwardIt>
size_t argmax(ForwardIt first, ForwardIt last) {
    return static_cast<size_t>(std::distance(first, std::max_element(first, last)));
}

} // namespace

Recognizer::Recognizer() = default;
Recognizer::~Recognizer() = default;

void Recognizer::setRecBatchNum(int n) {
    recBatchNum_ = n < 1 ? 1 : n;
}

void Recognizer::loadModel(const std::string& modelPath, bool useCuda,
                         bool useTensorrt, const std::string& trtCacheDir) {
    sessionOptions_.SetInterOpNumThreads(0);
    sessionOptions_.SetIntraOpNumThreads(0);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Height is fixed at kDstHeight=48; width and batch size vary (batching
    // up to recBatchNum_ crops per call — see getTextLines()). Opt/max
    // batch dim matches the configured batch so ORT TRT builds one engine
    // that covers the real runtime batch sizes instead of rebuilding
    // per-shape (default 6 matches PaddleOCR/RapidOCR and this project's
    // TENSORRT_ENGINE_PORT_PLAN.md Opsi 1 table).
    const int batch = recBatchNum_;
    const std::string minProfile = "x:1x3x48x32";
    const std::string optProfile = "x:" + std::to_string(batch) + "x3x48x320";
    const std::string maxProfile = "x:" + std::to_string(batch) + "x3x48x2048";
    detail::configureExecutionProviders(sessionOptions_, useCuda, useTensorrt, trtCacheDir,
        {minProfile, optProfile, maxProfile});

#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
#endif
    inputNamesPtr_ = detail::getInputNames(session_.get());
    outputNamesPtr_ = detail::getOutputNames(session_.get());
}

void Recognizer::finalizeKeys(std::vector<std::string> rawKeys) {
    if (rawKeys.empty()) {
        keys_.clear();
        return;
    }
    keys_.clear();
    keys_.reserve(rawKeys.size() + 2);
    keys_.emplace_back("#"); // CTC blank, index 0
    for (auto& k : rawKeys) keys_.push_back(std::move(k));
    keys_.emplace_back(" "); // trailing space, matches RapidOcrOnnx's convention
}

void Recognizer::loadKeysFromFile(const std::string& keysPath) {
    std::ifstream in(keysPath);
    if (!in.is_open()) {
        keys_.clear();
        return;
    }
    std::vector<std::string> rawKeys;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // CRLF fixtures on Windows
        rawKeys.push_back(line);
    }
    finalizeKeys(std::move(rawKeys));
}

bool Recognizer::loadKeysFromModelMetadata() {
    if (!session_) {
        return false;
    }
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session_->GetModelMetadata();
    std::string chars;
    try {
        Ort::AllocatedStringPtr charsPtr = metadata.LookupCustomMetadataMapAllocated("character", allocator);
        if (!charsPtr) return false;
        chars = charsPtr.get();
    } catch (const Ort::Exception&) {
        return false;
    }
    if (chars.empty()) {
        return false;
    }
    std::vector<std::string> rawKeys;
    std::istringstream iss(chars);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        rawKeys.push_back(line);
    }
    if (rawKeys.empty()) {
        return false;
    }
    finalizeKeys(std::move(rawKeys));
    return true;
}

RawTextLine Recognizer::scoreToTextLine(const float* outputData, size_t dataSize, size_t h, size_t w) const {
    auto keySize = keys_.size();
    std::string text;
    std::vector<float> scores;
    size_t lastIndex = 0;

    for (size_t i = 0; i < h; i++) {
        size_t start = i * w;
        size_t stop = (i + 1) * w;
        if (stop > dataSize) stop = dataSize; // guard against short buffers (differs from
                                               // RapidOcrOnnx's `dataSize - 1`, which can
                                               // under-read the last timestep by one class)
        if (start >= stop) continue;
        size_t maxIndex = argmax(outputData + start, outputData + stop);
        float maxValue = *std::max_element(outputData + start, outputData + stop);

        if (maxIndex > 0 && maxIndex < keySize && !(i > 0 && maxIndex == lastIndex)) {
            scores.push_back(maxValue);
            text += keys_[maxIndex];
        }
        lastIndex = maxIndex;
    }
    return {text, scores};
}

std::vector<float> Recognizer::buildBatchTensor(const std::vector<cv::Mat>& resizedCrops, int batchWidth) const {
    int batchSize = static_cast<int>(resizedCrops.size());
    // meanValues_/normValues_ are 3-element (BGR) — every crop must be
    // 3-channel for the per-channel plane math below to be valid. getTextLines()
    // only forwards 3-channel crops here (see its channel-count guard);
    // this is asserted rather than silently handled because a caller
    // reaching this private method with mixed channel counts is a bug in
    // this file, not user input to degrade gracefully on.
    const int channels = 3;
    std::vector<float> batchInput(static_cast<size_t>(batchSize) * channels * kDstHeight * batchWidth, 0.0f);
    // Build the batch tensor: each crop is normalized FIRST, then copied
    // into a zero-initialized batchWidth-wide slot. Padding is 0.0 in
    // NORMALIZED float space — NOT a pre-normalization pixel value (raw
    // pixel 0 would normalize to -1.0 under this project's mean/norm
    // constants, a different value). This exact order (normalize, then
    // pad) matches upstream PaddleOCR/RapidOCR's resize_norm_img.
    size_t cropPlaneSize = static_cast<size_t>(kDstHeight) * batchWidth; // per-channel plane size in the PADDED buffer
    size_t cropStride = cropPlaneSize * channels;

    for (int b = 0; b < batchSize; b++) {
        const cv::Mat& crop = resizedCrops[b];
        // Precondition: every crop must already be resized to exactly
        // kDstHeight rows and no wider than batchWidth — getTextLines()
        // guarantees this (it's the only real caller), but this is a
        // public-facing method via buildBatchTensorForTest() for testing,
        // so a violation is guarded rather than trusted: substractMeanNormalize's
        // output is sized crop.cols*crop.rows, and the row-copy loop below
        // indexes it assuming crop.rows == kDstHeight — get that wrong and
        // it's a heap out-of-bounds read, not just wrong output. Skipping
        // leaves this crop's slot as zero-padding (silent, not a crash).
        if (crop.rows != kDstHeight || crop.cols > batchWidth || crop.cols <= 0) {
            continue;
        }
        auto normalized = substractMeanNormalize(crop, meanValues_, normValues_); // CHW, row-major, crop.cols wide
        size_t normPlaneSize = static_cast<size_t>(crop.cols) * kDstHeight;
        float* dst = batchInput.data() + b * cropStride;
        // Copy ROW BY ROW, not as one contiguous block per channel: the
        // source plane is `kDstHeight` rows of `crop.cols` floats each
        // (packed, no gaps), but the destination plane is `kDstHeight` rows
        // of `batchWidth` floats each (crop.cols <= batchWidth, guaranteed
        // by the caller). A single contiguous copy would misalign every
        // row after the first whenever crop.cols != batchWidth — this was
        // an actual bug caught by real-model inference testing on Jetson
        // (produced garbled/wrong text), not just a theoretical concern.
        // test_recognizer.cpp now exercises this directly (no model needed)
        // so a regression here is caught by `ctest`, not just hardware runs.
        for (int ch = 0; ch < channels; ch++) {
            const float* srcPlane = normalized.data() + ch * normPlaneSize;
            float* dstPlane = dst + ch * cropPlaneSize;
            for (int row = 0; row < kDstHeight; row++) {
                std::copy(srcPlane + row * crop.cols, srcPlane + row * crop.cols + crop.cols,
                          dstPlane + row * batchWidth);
                // remaining (batchWidth - crop.cols) columns in this row
                // stay 0.0 from batchInput's initialization above.
            }
        }
    }

    return batchInput;
}

std::vector<RawTextLine> Recognizer::runBatchInference(const std::vector<cv::Mat>& resizedCrops, int batchWidth) {
    int batchSize = static_cast<int>(resizedCrops.size());
    std::vector<float> batchInput = buildBatchTensor(resizedCrops, batchWidth);

    std::array<int64_t, 4> inputShape{batchSize, 3, kDstHeight, batchWidth};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, batchInput.data(), batchInput.size(), inputShape.data(), inputShape.size());

    std::vector<const char*> inputNames = {inputNamesPtr_.front().get()};
    std::vector<const char*> outputNames = {outputNamesPtr_.front().get()};
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

    auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (outShape.size() < 3 || outShape[0] <= 0 || outShape[1] <= 0 || outShape[2] <= 0) {
        return std::vector<RawTextLine>(resizedCrops.size(), RawTextLine{"", {}});
        // guard: model output isn't the expected [batch, T, classes] shape
    }

    int64_t outBatch = outShape[0];
    int64_t timeSteps = outShape[1];
    int64_t numClasses = outShape[2];
    int64_t perItemCount = timeSteps * numClasses;
    int64_t outCount = outBatch * perItemCount;
    float* raw = outputTensors.front().GetTensorMutableData<float>();

    std::vector<RawTextLine> results;
    results.reserve(resizedCrops.size());
    // CTC-decode the full padded width per row, unconditionally — no
    // truncation/masking by each crop's real (unpadded) width. This relies
    // on the trained CRNN model emitting the CTC blank token in padding
    // columns; upstream does the same (confirmed by absence of masking
    // code there — see getTextLines() doc comment).
    for (int64_t b = 0; b < outBatch && b < static_cast<int64_t>(resizedCrops.size()); b++) {
        int64_t offset = b * perItemCount;
        if (offset + perItemCount > outCount) break; // guard: short buffer
        // Decode straight from this item's slice of the shared output
        // buffer — no per-item copy (numClasses is in the thousands for
        // PP-OCR's CJK dictionaries, so this is a real cost on the hot
        // path, not just style).
        results.push_back(scoreToTextLine(raw + offset, static_cast<size_t>(perItemCount),
                                           static_cast<size_t>(timeSteps), static_cast<size_t>(numClasses)));
    }
    while (results.size() < resizedCrops.size()) {
        results.push_back({"", {}}); // guard: model returned fewer rows than requested
    }
    return results;
}

std::vector<RawTextLine> Recognizer::runBatch(const std::vector<cv::Mat>& resizedCrops, int batchWidth) {
    try {
        return runBatchInference(resizedCrops, batchWidth);
    } catch (const std::exception&) {
        // Fallback for a non-standard custom model that doesn't accept
        // dynamic batch size (every official PP-OCR model does — see
        // TENSORRT_ENGINE_PORT_PLAN.md's documented rec profile,
        // opt=(6,3,48,320) — this path exists for robustness against
        // arbitrary user-supplied models, not because official models hit it).
        // Catches std::exception (not just Ort::Exception): runBatchInference
        // also runs buildBatchTensor(), pure computation that could throw
        // std::bad_alloc or similar — without this, such an error would
        // skip this per-crop fallback and fall through to the whole-page
        // catch in Engine::recognize(), discarding every line on the page
        // instead of just the crops in this one batch.
        std::vector<RawTextLine> results;
        results.reserve(resizedCrops.size());
        for (auto& crop : resizedCrops) {
            try {
                std::vector<cv::Mat> single{crop};
                auto oneResult = runBatchInference(single, crop.cols);
                results.push_back(oneResult.empty() ? RawTextLine{"", {}} : oneResult.front());
            } catch (const std::exception&) {
                results.push_back({"", {}}); // guard: this one crop failed even alone, skip it
            }
        }
        return results;
    }
}

std::vector<RawTextLine> Recognizer::getTextLines(std::vector<cv::Mat>& partImages) {
    size_t n = partImages.size();
    std::vector<RawTextLine> lines(n);
    if (!session_ || n == 0) {
        return lines; // guard: unloaded model, or nothing to do
    }

    // 1. Sort indices by ascending aspect ratio (width/height), same as
    // upstream — groups similarly-shaped crops together so each batch's
    // shared width wastes less padding. Degenerate (empty/zero-size) crops
    // sort first (ratio 0) and are filtered out per-item below, not batched.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        auto ratio = [](const cv::Mat& m) {
            return (m.empty() || m.rows <= 0) ? 0.0 : static_cast<double>(m.cols) / m.rows;
        };
        return ratio(partImages[a]) < ratio(partImages[b]);
    });

    const size_t batchNum = static_cast<size_t>(recBatchNum_ < 1 ? 1 : recBatchNum_);
    for (size_t groupStart = 0; groupStart < n; groupStart += batchNum) {
        size_t groupEnd = std::min(groupStart + batchNum, n);

        // 2. max_wh_ratio starts at the model's reference ratio (kRefImgWidth
        // / kDstHeight) and is raised to the widest crop's own ratio in this
        // batch — matches upstream exactly (see getTextLines() doc comment).
        double maxWhRatio = static_cast<double>(kRefImgWidth) / kDstHeight;
        std::vector<size_t> validIdx; // indices (into `order`) with a usable crop
        for (size_t k = groupStart; k < groupEnd; k++) {
            const cv::Mat& src = partImages[order[k]];
            if (src.empty() || src.rows <= 0 || src.channels() != 3) {
                lines[order[k]] = {"", {}}; // guard: degenerate crop upstream, or a
                continue;                   // custom-pipeline crop with an unsupported channel count
            }
            maxWhRatio = std::max(maxWhRatio, static_cast<double>(src.cols) / src.rows);
            validIdx.push_back(k);
        }
        if (validIdx.empty()) continue;

        int batchWidth = static_cast<int>(kDstHeight * maxWhRatio); // truncating cast, matches upstream int()

        // 3. Resize each valid crop to its own natural width (aspect-ratio
        // preserved, height fixed), capped at batchWidth — narrower crops
        // get zero-padded (in normalized space) inside runBatchInference().
        std::vector<cv::Mat> resizedCrops;
        resizedCrops.reserve(validIdx.size());
        for (size_t k : validIdx) {
            const cv::Mat& src = partImages[order[k]];
            float ratio = static_cast<float>(src.cols) / static_cast<float>(src.rows);
            int naturalWidth = static_cast<int>(std::ceil(kDstHeight * ratio));
            int resizedWidth = std::min(naturalWidth, batchWidth);
            resizedWidth = std::max(resizedWidth, 1); // guard: never resize to zero width
            cv::Mat resized;
            cv::resize(src, resized, cv::Size(resizedWidth, kDstHeight));
            resizedCrops.push_back(std::move(resized));
        }

        auto batchResults = runBatch(resizedCrops, batchWidth);
        for (size_t i = 0; i < validIdx.size() && i < batchResults.size(); i++) {
            lines[order[validIdx[i]]] = std::move(batchResults[i]);
        }
    }

    return lines;
}

} // namespace arbo::ocr
