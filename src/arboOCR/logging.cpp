#include "arboOCR/logging.hpp"

#include <iostream>
#include <mutex>

namespace arbo::ocr {

namespace {

std::mutex g_mu;
LogCallback g_callback;
LogLevel g_minLevel = LogLevel::Info;

const char* levelName(LogLevel level) {
    switch (level) {
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info:  return "INFO";
    case LogLevel::Warn:  return "WARN";
    case LogLevel::Error: return "ERROR";
    }
    return "INFO";
}

} // namespace

void setLogCallback(LogCallback callback) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_callback = std::move(callback);
}

void setMinLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_minLevel = level;
}

LogLevel minLogLevel() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_minLevel;
}

void log(LogLevel level, const std::string& message) {
    LogCallback cb;
    LogLevel minLevel;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!g_callback) return;
        if (static_cast<int>(level) < static_cast<int>(g_minLevel)) return;
        cb = g_callback;
        minLevel = g_minLevel;
    }
    (void)minLevel;
    try {
        cb(level, message);
    } catch (...) {
        // never let a broken sink crash OCR
    }
}

LogCallback makeStderrLogger() {
    return [](LogLevel level, const std::string& message) {
        std::cerr << "[arboOCR][" << levelName(level) << "] " << message << "\n";
    };
}

} // namespace arbo::ocr
