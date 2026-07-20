#pragma once

#include <string>
#include <vector>

namespace arbo::ocr {

struct DownloadResult {
    bool ok = false;
    std::string errorMessage;
    size_t bytesWritten = 0;
};

/// Download `url` to `destPath`. Skips (returns ok, 0 bytes) if the file
/// already exists and is non-empty. Never throws — unreachable hosts / HTTP
/// errors return ok=false with a message.
DownloadResult downloadFile(const std::string& url, const std::string& destPath);

/// Download the det/cls/rec ONNX files for `ocrVersion`/`modelType` from a
/// caller-supplied `baseUrl` (e.g. a HuggingFace/ModelScope/self-hosted
/// directory URL ending in '/'). arboOCR ships no default URL: PP-OCR model
/// hosting URLs are not stable/verified, so the caller decides the source.
/// Files are written as `<modelsDir>/<ocrVersion>_{det,cls}.onnx` and
/// `<modelsDir>/<ocrVersion>_rec_<modelType>.onnx`.
std::vector<DownloadResult> downloadOcrModels(
    const std::string& baseUrl,
    const std::string& ocrVersion,
    const std::string& modelType,
    const std::string& modelsDir
);

} // namespace arbo::ocr
