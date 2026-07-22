// tests/test_logging.cpp
#include <doctest/doctest.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "arboOCR/logging.hpp"

using namespace arbo::ocr;

namespace {

struct LogCapture {
    std::vector<std::pair<LogLevel, std::string>> lines;
    void install() {
        setLogCallback([this](LogLevel level, const std::string& msg) {
            lines.emplace_back(level, msg);
        });
    }
    void clear() { lines.clear(); }
};

// Restore silent default after each case so other tests stay quiet.
struct LogGuard {
    LogLevel prevMin = minLogLevel();
    ~LogGuard() {
        setLogCallback({});
        setMinLogLevel(prevMin);
    }
};

} // namespace

TEST_CASE("log is a no-op when no callback is set") {
    LogGuard guard;
    setLogCallback({});
    // must not throw / crash
    log(LogLevel::Error, "should be discarded");
}

TEST_CASE("log delivers messages at or above min level") {
    LogGuard guard;
    LogCapture cap;
    cap.install();
    setMinLogLevel(LogLevel::Info);

    log(LogLevel::Debug, "debug");
    log(LogLevel::Info, "info");
    log(LogLevel::Warn, "warn");
    log(LogLevel::Error, "error");

    REQUIRE(cap.lines.size() == 3);
    CHECK(cap.lines[0].first == LogLevel::Info);
    CHECK(cap.lines[0].second == "info");
    CHECK(cap.lines[1].first == LogLevel::Warn);
    CHECK(cap.lines[2].first == LogLevel::Error);
}

TEST_CASE("setMinLogLevel Debug passes all levels") {
    LogGuard guard;
    LogCapture cap;
    cap.install();
    setMinLogLevel(LogLevel::Debug);

    log(LogLevel::Debug, "d");
    log(LogLevel::Error, "e");
    REQUIRE(cap.lines.size() == 2);
    CHECK(cap.lines[0].second == "d");
    CHECK(cap.lines[1].second == "e");
}

TEST_CASE("callback exceptions are swallowed") {
    LogGuard guard;
    setLogCallback([](LogLevel, const std::string&) {
        throw std::runtime_error("sink broken");
    });
    log(LogLevel::Error, "must not escape");
}

TEST_CASE("makeStderrLogger is non-null") {
    auto cb = makeStderrLogger();
    CHECK(static_cast<bool>(cb));
}
