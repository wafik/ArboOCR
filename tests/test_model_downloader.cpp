#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include "arboOCR/model_downloader.hpp"

using namespace arbo::ocr;
namespace fs = std::filesystem;

TEST_CASE("downloadFile skips download if destination already exists and is non-empty") {
    fs::create_directories("tests/fixtures/downloader_tmp");
    std::string dest = "tests/fixtures/downloader_tmp/already_here.txt";
    {
        std::ofstream f(dest);
        f << "existing content";
    }
    auto result = downloadFile("https://example.invalid/should-not-be-fetched", dest);
    CHECK(result.ok == true);
    CHECK(result.bytesWritten == 0);
    fs::remove(dest);
}

TEST_CASE("downloadFile with an unreachable host returns ok=false, does not throw") {
    std::string dest = "tests/fixtures/downloader_tmp/unreachable.txt";
    auto result = downloadFile("https://this-host-does-not-exist.invalid/file.onnx", dest);
    CHECK(result.ok == false);
    CHECK_FALSE(result.errorMessage.empty());
    CHECK_FALSE(fs::exists(dest));
}

TEST_CASE("downloadOcrModels covers four files including the optional rec dict") {
    auto names = ocrModelFileNames("PP-OCRv5", "mobile");
    REQUIRE(names.size() == 4);
    CHECK(names[0] == "PP-OCRv5_det.onnx");
    CHECK(names[1] == "PP-OCRv5_cls.onnx");
    CHECK(names[2] == "PP-OCRv5_rec_mobile.onnx");
    CHECK(names[3] == "PP-OCRv5_rec_mobile_dict.txt");

    // Unreachable host: no network needed, every fetch fails the same way.
    auto results = downloadOcrModels(
        "https://this-host-does-not-exist.invalid/models",
        "PP-OCRv5", "mobile", "tests/fixtures/downloader_tmp/models");
    REQUIRE(results.size() == 4);
    for (auto& r : results) CHECK(r.ok == false);
    // The dict failure is flagged as possibly ignorable; the models are not.
    CHECK(results.back().errorMessage.find("optional") != std::string::npos);
    CHECK(results[2].errorMessage.find("optional") == std::string::npos);
    fs::remove_all("tests/fixtures/downloader_tmp/models");
}
