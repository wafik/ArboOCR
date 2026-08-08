#include "arboOCR/model_downloader.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <curl/curl.h>

#ifdef _WIN32
#include <process.h>
#define ARBOOCR_GETPID _getpid
#else
#include <unistd.h>
#define ARBOOCR_GETPID getpid
#endif

namespace fs = std::filesystem;

namespace arbo::ocr {

namespace {

size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* out = static_cast<std::ofstream*>(userp);
    size_t total = size * nmemb;
    out->write(static_cast<char*>(contents), static_cast<std::streamsize>(total));
    return total;
}

const char* envOrNull(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

// ─── SHA-256 (FIPS 180-4) ────────────────────────────────────────────────────
// ponytail: vendored rather than linking OpenSSL. This is the only hash
// arboOCR needs, libcurl exposes none, and zlib gives CRC32 only — a whole
// crypto dependency in vcpkg.json for ~70 lines is a bad trade. Swap in
// EVP_Digest if arboOCR ever needs a second algorithm.

constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t block[64] = {};
    size_t blockLen = 0;
    uint64_t totalBits = 0;

    void compress() {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24)
                 | (static_cast<uint32_t>(block[i * 4 + 1]) << 16)
                 | (static_cast<uint32_t>(block[i * 4 + 2]) << 8)
                 | static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + kRoundConstants[i] + w[i];
            uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    // Feeds bytes through the block buffer without touching the length
    // counter, so finish() can append padding that is not message content.
    void absorb(const uint8_t* p, size_t n) {
        while (n > 0) {
            // Parenthesized: curl.h drags in windows.h, whose min() macro
            // would otherwise eat the call.
            size_t take = (std::min)(n, size_t(64) - blockLen);
            std::memcpy(block + blockLen, p, take);
            blockLen += take;
            p += take;
            n -= take;
            if (blockLen == 64) {
                compress();
                blockLen = 0;
            }
        }
    }

    void update(const uint8_t* p, size_t n) {
        totalBits += static_cast<uint64_t>(n) * 8;
        absorb(p, n);
    }

    std::string finish() {
        const uint64_t bits = totalBits;
        const uint8_t padStart = 0x80;
        absorb(&padStart, 1);
        const uint8_t zero = 0;
        while (blockLen != 56) absorb(&zero, 1);
        uint8_t lengthBE[8];
        for (int i = 0; i < 8; ++i) {
            lengthBE[i] = static_cast<uint8_t>((bits >> (56 - i * 8)) & 0xff);
        }
        absorb(lengthBE, 8);

        static const char* digits = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (uint32_t word : h) {
            for (int i = 3; i >= 0; --i) {
                uint8_t byte = static_cast<uint8_t>((word >> (i * 8)) & 0xff);
                out.push_back(digits[byte >> 4]);
                out.push_back(digits[byte & 0x0f]);
            }
        }
        return out;
    }
};

// ─── Pinned manifest ─────────────────────────────────────────────────────────
// Hashes of the assets published at defaultModelsTag(). Regenerate with
// scripts/gen_model_manifest.ps1 when cutting a new models tag; the tag and
// these hashes must move together or the download hard-fails, which is the
// intended failure mode.

struct ManifestEntry {
    const char* name;
    const char* sha256;
};

constexpr ManifestEntry kManifest[] = {
    {"PP-OCRv6_cls.onnx",            "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"},
    {"PP-OCRv6_det.onnx",            "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5"},
    {"PP-OCRv6_det_medium.onnx",     "92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2"},
    {"PP-OCRv6_det_small.onnx",      "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"},
    {"PP-OCRv6_det_tiny.onnx",       "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5"},
    {"PP-OCRv6_rec_medium.onnx",     "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b"},
    {"PP-OCRv6_rec_medium_dict.txt", "f7aa897ca828a4c7c9e2739c30f9161a33306d532f020bcdb91dcfb664a5507e"},
    {"PP-OCRv6_rec_small.onnx",      "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"},
    {"PP-OCRv6_rec_small_dict.txt",  "f7aa897ca828a4c7c9e2739c30f9161a33306d532f020bcdb91dcfb664a5507e"},
    {"PP-OCRv6_rec_tiny.onnx",       "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b"},
    {"PP-OCRv6_rec_tiny_dict.txt",   "34d139222b6e8d84830f57476e39f01e76be3baffde3e9df75a079e251140598"},
};

} // namespace

const char* defaultModelsTag() { return "models-v1"; }

std::string defaultModelsBaseUrl() {
    if (const char* override = envOrNull("ARBOOCR_MODELS_URL")) {
        std::string url(override);
        if (!url.empty() && url.back() != '/') url.push_back('/');
        return url;
    }
    return std::string("https://github.com/ARBO-TEAM/arbo-ocr-models/releases/download/")
        + defaultModelsTag() + "/";
}

std::string defaultModelsCacheDir() {
    fs::path root;
    if (const char* override = envOrNull("ARBOOCR_CACHE_DIR")) {
        root = override;
        return (root / defaultModelsTag()).string();
    }
#if defined(_WIN32)
    if (const char* localAppData = envOrNull("LOCALAPPDATA")) {
        root = localAppData;
    } else if (const char* userProfile = envOrNull("USERPROFILE")) {
        root = fs::path(userProfile) / "AppData" / "Local";
    }
#elif defined(__APPLE__)
    if (const char* home = envOrNull("HOME")) {
        root = fs::path(home) / "Library" / "Caches";
    }
#else
    if (const char* xdg = envOrNull("XDG_CACHE_HOME")) {
        root = xdg;
    } else if (const char* home = envOrNull("HOME")) {
        root = fs::path(home) / ".cache";
    }
#endif
    if (root.empty()) {
        std::error_code ec;
        root = fs::temp_directory_path(ec);
        if (ec) root = ".";
    }
    return (root / "arboOCR" / "models" / defaultModelsTag()).string();
}

std::string sha256File(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    Sha256 ctx;
    std::vector<char> buffer(64 * 1024);
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got <= 0) break;
        ctx.update(reinterpret_cast<const uint8_t*>(buffer.data()),
                   static_cast<size_t>(got));
    }
    return ctx.finish();
}

std::string knownSha256(const std::string& fileName) {
    for (const auto& entry : kManifest) {
        if (fileName == entry.name) return entry.sha256;
    }
    return "";
}

DownloadResult downloadFile(
    const std::string& url, const std::string& destPath, const std::string& expectedSha256
) {
    fs::path dest(destPath);
    std::error_code ec;

    // Already-present fast path. With a pinned hash this is a real check
    // rather than a guess: a truncated or tampered file is removed and
    // refetched instead of being handed to ONNX Runtime.
    if (fs::exists(dest, ec) && fs::file_size(dest, ec) > 0) {
        if (expectedSha256.empty() || sha256File(destPath) == expectedSha256) {
            return {true, "", 0};
        }
        fs::remove(dest, ec);
    }

    fs::create_directories(dest.parent_path(), ec);

    // ponytail: pid alone makes the temp name unique across processes, which
    // is the race that actually happens (two workers starting at once). Two
    // threads in ONE process downloading the same URL would still collide —
    // add a counter if that ever becomes real.
    fs::path tmp = dest;
    tmp += "." + std::to_string(ARBOOCR_GETPID()) + ".tmp";

    std::ofstream out(tmp, std::ios::binary);
    if (!out.is_open()) {
        return {false, "cannot open destination file for writing: " + tmp.string(), 0};
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        out.close();
        fs::remove(tmp, ec);
        return {false, "curl_easy_init failed", 0};
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "arboOCR/model_downloader");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    CURLcode res = curl_easy_perform(curl);
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    curl_easy_cleanup(curl);
    out.close();

    if (res != CURLE_OK || httpCode >= 400) {
        fs::remove(tmp, ec);
        std::string err = (res != CURLE_OK)
            ? std::string("curl error: ") + curl_easy_strerror(res)
            : "HTTP " + std::to_string(httpCode);
        return {false, err, 0};
    }

    if (!expectedSha256.empty()) {
        std::string actual = sha256File(tmp.string());
        if (actual != expectedSha256) {
            fs::remove(tmp, ec);
            return {false,
                    "checksum mismatch for " + url + ": expected " + expectedSha256
                        + ", got " + (actual.empty() ? "<unreadable>" : actual),
                    0};
        }
    }

    size_t bytes = fs::exists(tmp, ec) ? static_cast<size_t>(fs::file_size(tmp, ec)) : 0;

    // Publish atomically. std::filesystem::rename replaces an existing
    // destination on both POSIX and Win32 (MoveFileEx + REPLACE_EXISTING), so
    // no reader ever sees a partial file.
    fs::rename(tmp, dest, ec);
    if (ec) {
        std::error_code cleanup;
        fs::remove(tmp, cleanup);
        return {false, "cannot publish download to " + destPath + ": " + ec.message(), 0};
    }
    return {true, "", bytes};
}

std::vector<std::string> ocrModelFileNames(
    const std::string& ocrVersion, const std::string& modelType
) {
    std::string recStem = ocrVersion + "_rec_" + modelType;
    return {
        ocrVersion + "_det.onnx",
        ocrVersion + "_cls.onnx",
        recStem + ".onnx",
        recStem + "_dict.txt",
    };
}

// The dict is downloaded alongside the three ONNX files because
// resolveModelPaths() expects it, but it is legitimately optional: a rec model
// that embeds its charset in the ONNX "character" metadata key never needs one,
// and such repos do not host a dict at all. Rather than introduce a second
// result type for "failed but that's fine", the dict keeps the same
// DownloadResult shape and just carries an errorMessage saying the failure may
// be ignorable — the caller still sees ok=false and is not lied to.
std::vector<DownloadResult> downloadOcrModels(
    const std::string& baseUrl, const std::string& ocrVersion,
    const std::string& modelType, const std::string& modelsDir
) {
    fs::path dir(modelsDir);
    std::string base = baseUrl.empty() ? defaultModelsBaseUrl() : baseUrl;
    if (!base.empty() && base.back() != '/') base.push_back('/');

    std::vector<DownloadResult> results;
    for (auto& name : ocrModelFileNames(ocrVersion, modelType)) {
        // knownSha256 is looked up by name, so a custom mirror still gets
        // integrity checking for stock file names, and unknown names (custom
        // or fine-tuned weights) fall back to unverified as before.
        results.push_back(downloadFile(base + name, (dir / name).string(), knownSha256(name)));
    }
    if (!results.back().ok) {
        results.back().errorMessage +=
            " (optional: the dict is only needed when the rec model does not"
            " embed its charset in the ONNX \"character\" metadata key)";
    }
    return results;
}

} // namespace arbo::ocr
