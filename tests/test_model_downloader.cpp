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

namespace {

std::string hashOf(const std::string& content) {
    fs::create_directories("tests/fixtures/downloader_tmp");
    const std::string path = "tests/fixtures/downloader_tmp/sha_input.bin";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    std::string digest = sha256File(path);
    fs::remove(path);
    return digest;
}

} // namespace

TEST_CASE("sha256File matches the NIST vectors, including the padding boundary") {
    // Empty input: the pad byte plus the length field must still produce one
    // full block.
    CHECK(hashOf("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hashOf("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // Exactly 56 bytes — the length field no longer fits after the 0x80 pad,
    // so this must spill into a second block. The case a hand-rolled SHA-256
    // gets wrong.
    CHECK(hashOf("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
          == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // 100 kB: larger than the 64 kB read buffer, so this also covers the
    // streaming loop feeding the hash across more than one read().
    CHECK(hashOf(std::string(100000, 'a'))
          == "6d1cf22d7cc09b085dfc25ee1a1f3ae0265804c607bc2074ad253bcc82fd81ee");
    CHECK(sha256File("tests/fixtures/downloader_tmp/definitely-not-here").empty());
}

TEST_CASE("downloadFile refuses to keep a file whose checksum does not match") {
    fs::create_directories("tests/fixtures/downloader_tmp");
    const std::string dest = "tests/fixtures/downloader_tmp/corrupt.onnx";
    {
        std::ofstream f(dest, std::ios::binary);
        f << "truncated garbage that is not the model";
    }
    const std::string wantedHash(64, 'a');

    // Non-empty destination + wrong hash: the skip-if-exists fast path must not
    // fire. The host is unreachable, so the refetch fails and we are left with
    // no file at all rather than a corrupt one that ORT would mmap.
    auto result = downloadFile("https://this-host-does-not-exist.invalid/m.onnx", dest, wantedHash);
    CHECK(result.ok == false);
    CHECK_FALSE(fs::exists(dest));

    // And no temp file survives the failure.
    int leftovers = 0;
    for (const auto& e : fs::directory_iterator("tests/fixtures/downloader_tmp")) {
        if (e.path().extension() == ".tmp") ++leftovers;
    }
    CHECK(leftovers == 0);
}

TEST_CASE("knownSha256 pins the stock model names and nothing else") {
    CHECK(knownSha256("PP-OCRv6_rec_small.onnx").size() == 64);
    CHECK(knownSha256("PP-OCRv6_det.onnx").size() == 64);
    CHECK(knownSha256("my-finetuned-rec.onnx").empty());
}

TEST_CASE("defaultModelsCacheDir is tag-scoped so a new tag cannot reuse old files") {
    const std::string dir = defaultModelsCacheDir();
    CHECK(dir.find(defaultModelsTag()) != std::string::npos);
    CHECK(defaultModelsBaseUrl().find(defaultModelsTag()) != std::string::npos);
    CHECK(defaultModelsBaseUrl().back() == '/');
}
