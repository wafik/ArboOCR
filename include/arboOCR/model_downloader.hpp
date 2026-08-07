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

/// The four file names `downloadOcrModels` fetches, in order:
///   `<ocrVersion>_det.onnx`, `<ocrVersion>_cls.onnx`,
///   `<ocrVersion>_rec_<modelType>.onnx`, `<ocrVersion>_rec_<modelType>_dict.txt`.
/// Both the URL and the destination path are derived from these names.
std::vector<std::string> ocrModelFileNames(
    const std::string& ocrVersion,
    const std::string& modelType
);

/// Download the det/cls/rec ONNX files *and* the rec dict for
/// `ocrVersion`/`modelType` from a caller-supplied `baseUrl` (e.g. a
/// HuggingFace/ModelScope/self-hosted directory URL ending in '/'). arboOCR
/// ships no default URL: PP-OCR model hosting URLs are not stable/verified, so
/// the caller decides the source. Files are written as
/// `<modelsDir>/<name>` for each name in `ocrModelFileNames()`, so the result
/// vector always has four entries in that same order.
/// The dict (4th entry) is best-effort: a rec model that embeds its charset in
/// the ONNX `character` metadata key needs no dict file, so a failure there is
/// returned as ok=false with an errorMessage that says it may be ignorable —
/// it is not promoted to an error for the whole call.
std::vector<DownloadResult> downloadOcrModels(
    const std::string& baseUrl,
    const std::string& ocrVersion,
    const std::string& modelType,
    const std::string& modelsDir
);

} // namespace arbo::ocr
