#pragma once
// Optional structured logging for embedders. Default is silent (no
// callback): the library never prints to stdout/stderr. Install a
// callback to route messages into spdlog, glog, your service logger, etc.

#include <functional>
#include <string>

namespace arbo::ocr {

enum class LogLevel {
    Debug = 0,
    Info  = 1,
    Warn  = 2,
    Error = 3,
};

/// (level, message). Must be safe to call from any thread that runs
/// Engine/Detector/Recognizer; callers who share state across threads
/// should synchronize inside their own callback.
using LogCallback = std::function<void(LogLevel level, const std::string& message)>;

/// Install or clear (pass nullptr / empty) the process-wide log sink.
void setLogCallback(LogCallback callback);

/// Messages below this level are dropped before the callback is invoked.
/// Default: Info (Debug is filtered out unless lowered).
void setMinLogLevel(LogLevel level);
LogLevel minLogLevel();

/// Emit a log line if a callback is set and level >= minLogLevel().
/// Never throws. Safe when no callback is installed (no-op).
void log(LogLevel level, const std::string& message);

/// Convenience: write level-prefixed lines to stderr. Useful for demos.
LogCallback makeStderrLogger();

} // namespace arbo::ocr
