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
