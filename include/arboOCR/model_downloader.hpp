#pragma once

#include <string>
#include <vector>

namespace arbo::ocr {

struct DownloadResult {
    bool ok = false;
    std::string errorMessage;
    size_t bytesWritten = 0;
};

/// Release tag the built-in checksum manifest describes. The tag — not a
/// branch — is what makes the default URL safe to hardcode: release assets
/// under a tag are immutable, so a cached file that matches the manifest is
/// the same file every future version of arboOCR will expect.
const char* defaultModelsTag();

/// Directory URL (trailing '/') for `defaultModelsTag()`'s release assets.
/// Overridden by the `ARBOOCR_MODELS_URL` environment variable, for internal
/// mirrors and air-gapped installs.
std::string defaultModelsBaseUrl();

/// Where auto-downloaded weights land, tag-scoped so a future `models-v2`
/// never reuses a `models-v1` file:
///   Windows  %LOCALAPPDATA%\arboOCR\models\<tag>
///   macOS    ~/Library/Caches/arboOCR/models/<tag>
///   Linux    $XDG_CACHE_HOME (or ~/.cache)/arboOCR/models/<tag>
/// `ARBOOCR_CACHE_DIR` overrides the root. Falls back to the system temp
/// directory when no home directory is discoverable.
std::string defaultModelsCacheDir();

/// Lowercase hex SHA-256 of the file at `path`; empty string if unreadable.
std::string sha256File(const std::string& path);

/// Pinned SHA-256 for a stock model file name at `defaultModelsTag()`, or an
/// empty string for a name that is not in the manifest (custom or fine-tuned
/// weights, or a different `ocrVersion`).
std::string knownSha256(const std::string& fileName);

/// Download `url` to `destPath`.
///
/// When `expectedSha256` is non-empty the bytes are verified before they are
/// published: the body is written to a sibling `.<pid>.tmp` file, hashed, and
/// only then renamed onto `destPath`. A reader therefore never observes a
/// partial or wrong-hash file, and two processes racing the same model cannot
/// corrupt each other's write. A destination that already exists is re-hashed
/// and re-fetched if it does not match — a truncated `.onnx` gets repaired
/// rather than being mmap'd into ONNX Runtime.
///
/// With an empty `expectedSha256` an existing non-empty destination is
/// skipped unverified (returns ok, 0 bytes), which is all that can be checked
/// without a pinned hash.
///
/// Never throws — unreachable hosts, HTTP errors, and checksum mismatches all
/// return ok=false with a message.
DownloadResult downloadFile(
    const std::string& url,
    const std::string& destPath,
    const std::string& expectedSha256 = ""
);

/// The four file names `downloadOcrModels` fetches, in order:
///   `<ocrVersion>_det.onnx`, `<ocrVersion>_cls.onnx`,
///   `<ocrVersion>_rec_<modelType>.onnx`, `<ocrVersion>_rec_<modelType>_dict.txt`.
/// Both the URL and the destination path are derived from these names.
std::vector<std::string> ocrModelFileNames(
    const std::string& ocrVersion,
    const std::string& modelType
);

/// Download the det/cls/rec ONNX files *and* the rec dict for
/// `ocrVersion`/`modelType` into `<modelsDir>/<name>` for each name in
/// `ocrModelFileNames()`, so the result vector always has four entries in that
/// same order.
///
/// An empty `baseUrl` uses `defaultModelsBaseUrl()`. Pass your own directory
/// URL (ending in '/') to fetch from an internal mirror or your own artifact
/// store instead; checksums are still enforced for any file name the built-in
/// manifest knows, so a mirror serving different bytes is rejected rather than
/// loaded.
///
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
