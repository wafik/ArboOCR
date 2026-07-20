#include "arboOCR/model_downloader.hpp"

#include <filesystem>
#include <fstream>

#include <curl/curl.h>

namespace fs = std::filesystem;

namespace arbo::ocr {

namespace {

size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* out = static_cast<std::ofstream*>(userp);
    size_t total = size * nmemb;
    out->write(static_cast<char*>(contents), static_cast<std::streamsize>(total));
    return total;
}

} // namespace

DownloadResult downloadFile(const std::string& url, const std::string& destPath) {
    fs::path dest(destPath);
    if (fs::exists(dest) && fs::file_size(dest) > 0) {
        return {true, "", 0};
    }

    fs::create_directories(dest.parent_path());
    std::ofstream out(dest, std::ios::binary);
    if (!out.is_open()) {
        return {false, "cannot open destination file for writing: " + destPath, 0};
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
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
        fs::remove(dest);
        std::string err = (res != CURLE_OK)
            ? std::string("curl error: ") + curl_easy_strerror(res)
            : "HTTP " + std::to_string(httpCode);
        return {false, err, 0};
    }

    size_t bytes = fs::exists(dest) ? fs::file_size(dest) : 0;
    return {true, "", bytes};
}

std::vector<DownloadResult> downloadOcrModels(
    const std::string& baseUrl, const std::string& ocrVersion,
    const std::string& modelType, const std::string& modelsDir
) {
    fs::path dir(modelsDir);
    std::string base = baseUrl;
    if (!base.empty() && base.back() != '/') base.push_back('/');

    std::vector<std::pair<std::string, std::string>> files = {
        {base + ocrVersion + "_det.onnx", (dir / (ocrVersion + "_det.onnx")).string()},
        {base + ocrVersion + "_cls.onnx", (dir / (ocrVersion + "_cls.onnx")).string()},
        {base + ocrVersion + "_rec_" + modelType + ".onnx", (dir / (ocrVersion + "_rec_" + modelType + ".onnx")).string()},
    };

    std::vector<DownloadResult> results;
    for (auto& [url, dest] : files) {
        results.push_back(downloadFile(url, dest));
    }
    return results;
}

} // namespace arbo::ocr
