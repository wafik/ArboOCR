// cli/arboocr_demo.cpp — minimal arboOCR quickstart: recognize one image, or a
// whole list of them against a single loaded Engine (--images-from).
#include <filesystem>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include <opencv2/imgcodecs.hpp>

#include "arboOCR/engine.hpp"
#include "arboOCR/logging.hpp"
#include "arboOCR/markdown.hpp"
#include "arboOCR/model_downloader.hpp"
#include "arboOCR/visualize.hpp"

namespace {

// Appended to --help: wrappers drive this binary over subprocess and branch on
// the exit code, so the batch rule has to be written down somewhere they read.
constexpr const char* kExitCodesHelp =
    "Exit codes:\n"
    "  0  success — every image produced at least one text line\n"
    "  1  usage error, or no text found (single image: the one image was empty;\n"
    "     --images-from: at least one image was empty — a partial batch)\n"
    "  2  nothing usable ran — model load failed, recognition threw, or the\n"
    "     --images-from list could not be opened\n";

// A file of paths (one per line) rather than a directory glob or a repeatable
// --image: no glob library and no recursion policy to decide, it composes with
// find/ls/Get-ChildItem, and it has no command-line length ceiling (Windows
// caps argv near 32k, which a repeatable flag would hit around 200 paths).
// The caller decides what the set is.
std::vector<std::string> readImageList(std::istream& in) {
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(in, line)) {
        // Trailing \r so a CRLF list written on Windows works anywhere.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        paths.push_back(line);
    }
    return paths;
}

} // namespace

int main(int argc, char* argv[]) {
    // Library default is silent (no callback). --log-level opts in below.
    cxxopts::Options opts("arboocr_demo", "Recognize text in one or many images with arboOCR");
    opts.add_options()
        ("image", "Path to input image", cxxopts::value<std::string>())
        ("images-from", "Read image paths from this file, one per line (\"-\" reads stdin); blank"
                        " lines and lines starting with # are skipped. All of them share one"
                        " loaded Engine. Mutually exclusive with --image",
            cxxopts::value<std::string>())
        ("models-dir", "Directory containing ONNX models", cxxopts::value<std::string>()->default_value("models"))
        ("ocr-version", "OCR model version", cxxopts::value<std::string>()->default_value("PP-OCRv6"))
        ("model-type", "Recognizer size — tiny/small/medium (default small; detector is always the same file unless --det-model)",
            cxxopts::value<std::string>()->default_value("small"))
        ("no-download", "Never fetch missing models; fail instead (same as ARBOOCR_OFFLINE=1)",
            cxxopts::value<bool>()->default_value("false"))
        ("models-url", "Directory URL to fetch missing models from (default: the pinned arboOCR models release)",
            cxxopts::value<std::string>()->default_value(""))
        ("download-models", "Fetch the models for --ocr-version/--model-type into the cache and exit",
            cxxopts::value<bool>()->default_value("false"))
        // Detection tuning (accuracy knobs — see EngineConfig in engine.hpp).
        ("det-limit-side-len", "Longest image side for detection resize",
            cxxopts::value<int>()->default_value("960"))
        ("det-thresh", "Detection probability-map binarization threshold",
            cxxopts::value<float>()->default_value("0.3"))
        ("det-box-thresh", "Minimum mean score for a detected box to be kept",
            cxxopts::value<float>()->default_value("0.5"))
        ("det-unclip-ratio", "Box expansion ratio applied after binarization",
            cxxopts::value<float>()->default_value("1.6"))
        ("split-overmerged", "Split wide det boxes that fuse two side-by-side fields (ink-gap heuristic)",
            cxxopts::value<bool>()->default_value("false"))
        // Recognition tuning.
        ("rec-batch-num", "Crops per recognition inference call (clamped to >= 1)",
            cxxopts::value<int>()->default_value("6"))
        ("min-confidence", "Drop lines below this recognition confidence (0 disables filtering)",
            cxxopts::value<float>()->default_value("0.5"))
        ("angle", "Enable angle classification", cxxopts::value<bool>()->default_value("false"))
        ("cuda", "Request CUDA execution provider", cxxopts::value<bool>()->default_value("false"))
        ("tensorrt", "Request TensorRT execution provider", cxxopts::value<bool>()->default_value("false"))
        ("fp16", "TensorRT FP16 (default true; only used with --tensorrt)",
            cxxopts::value<bool>()->default_value("true"))
        ("trt-cache-dir", "Directory for cached TensorRT engines (only used with --tensorrt)",
            cxxopts::value<std::string>()->default_value("models/trt_engines"))
        ("clahe", "Apply CLAHE contrast enhancement before detection (helps low-contrast/faded documents)",
            cxxopts::value<bool>()->default_value("false"))
        ("json", "Print machine-readable JSON (only JSON on stdout; suppresses the human-readable lines)",
            cxxopts::value<bool>()->default_value("false"))
        ("log-level", "Log engine events to stderr at this level — debug|info|warn|error (default: silent)",
            cxxopts::value<std::string>())
        ("draw", "Write a copy of the image with detected boxes outlined to this path",
            cxxopts::value<std::string>())
        ("markdown", "Write the reconstructed markdown document to this path (implies --word-boxes)",
            cxxopts::value<std::string>())
        ("word-boxes", "Also emit a polygon per word (per character for CJK)",
            cxxopts::value<bool>()->default_value("false"))
        ("det-model", "Override detector ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("cls-model", "Override classifier ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("rec-model", "Override recognizer ONNX path", cxxopts::value<std::string>()->default_value(""))
        ("dict", "Override character dict path (fallback if no ONNX metadata)",
            cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = opts.parse(argc, argv);
    } catch (const std::exception& e) {
        // Uncaught, this exception unwinds past main() and hits
        // std::terminate() — which this toolchain's hardened runtime turns
        // into an unhelpful STATUS_STACK_BUFFER_OVERRUN crash instead of a
        // clean error (e.g. any unrecognized flag). Fail cleanly instead.
        std::cerr << "arboocr_demo: " << e.what() << "\n\n" << opts.help() << std::endl;
        return 1;
    }
    const bool batchMode = result.count("images-from") > 0;
    // Asking for help is not a usage error: --help exits 0 so `cmd --help` in a
    // CI check or a wrapper's install probe succeeds. Giving no input at all
    // still prints the same text and exits 1.
    if (result.count("help")) {
        std::cout << opts.help() << "\n" << kExitCodesHelp << std::endl;
        return 0;
    }
    // --download-models is a prefetch mode: it fetches and exits, so requiring
    // an image to name a file it will never open would be nonsense.
    const bool prefetchOnly = result["download-models"].as<bool>();
    if (!result.count("image") && !batchMode && !prefetchOnly) {
        std::cout << opts.help() << "\n" << kExitCodesHelp << std::endl;
        return 1;
    }
    if (result.count("image") && batchMode) {
        std::cerr << "arboocr_demo: --image and --images-from are mutually exclusive"
                     " (pass one image, or a list of them — not both)" << std::endl;
        return 1;
    }
    // ponytail: --draw and --markdown each take a single output path, so N images
    // would write N times over one file. Refusing beats silently keeping only the
    // last page. Upgrade path if anyone actually needs it: an output-directory flag.
    if (batchMode && result.count("draw")) {
        std::cerr << "arboocr_demo: --draw cannot be combined with --images-from"
                     " (single output path for many images)" << std::endl;
        return 1;
    }
    if (batchMode && result.count("markdown")) {
        std::cerr << "arboocr_demo: --markdown cannot be combined with --images-from"
                     " (single output path for many images)" << std::endl;
        return 1;
    }

    // Opt-in logging: without --log-level the library stays silent (no callback).
    if (result.count("log-level")) {
        const std::string level = result["log-level"].as<std::string>();
        arbo::ocr::LogLevel minLevel = arbo::ocr::LogLevel::Info;
        if (level == "debug") {
            minLevel = arbo::ocr::LogLevel::Debug;
        } else if (level == "info") {
            minLevel = arbo::ocr::LogLevel::Info;
        } else if (level == "warn") {
            minLevel = arbo::ocr::LogLevel::Warn;
        } else if (level == "error") {
            minLevel = arbo::ocr::LogLevel::Error;
        } else {
            std::cerr << "arboocr_demo: unknown --log-level \"" << level
                      << "\" (expected one of: debug, info, warn, error)" << std::endl;
            return 1;
        }
        arbo::ocr::setLogCallback(arbo::ocr::makeStderrLogger());
        arbo::ocr::setMinLogLevel(minLevel);
    }

    arbo::ocr::EngineConfig cfg;
    cfg.modelsDir = result["models-dir"].as<std::string>();
    cfg.ocrVersion = result["ocr-version"].as<std::string>();
    cfg.modelType = result["model-type"].as<std::string>();
    cfg.detLimitSideLen = result["det-limit-side-len"].as<int>();
    cfg.detThresh = result["det-thresh"].as<float>();
    cfg.detBoxThresh = result["det-box-thresh"].as<float>();
    cfg.detUnclipRatio = result["det-unclip-ratio"].as<float>();
    cfg.splitOvermerged = result["split-overmerged"].as<bool>();
    cfg.recBatchNum = result["rec-batch-num"].as<int>();
    cfg.minimumConfidence = result["min-confidence"].as<float>();
    cfg.useAngleCls = result["angle"].as<bool>();
    cfg.useCuda = result["cuda"].as<bool>();
    cfg.useTensorrt = result["tensorrt"].as<bool>();
    cfg.useFp16 = result["fp16"].as<bool>();
    cfg.trtCacheDir = result["trt-cache-dir"].as<std::string>();
    cfg.useClahe = result["clahe"].as<bool>();
    // --markdown detects `key | value` rows from the wide gap between words, so
    // it needs word boxes even when --word-boxes was not asked for.
    cfg.returnWordBoxes = result["word-boxes"].as<bool>() || result.count("markdown") > 0;
    cfg.detModelPath = result["det-model"].as<std::string>();
    cfg.clsModelPath = result["cls-model"].as<std::string>();
    cfg.recModelPath = result["rec-model"].as<std::string>();
    cfg.dictPath = result["dict"].as<std::string>();
    cfg.autoDownload = !result["no-download"].as<bool>();
    cfg.modelsBaseUrl = result["models-url"].as<std::string>();

    // Prefetch-and-exit: warms the cache in CI or a Docker build layer so the
    // first real run does not pay for ~22 MB mid-request.
    if (prefetchOnly) {
        if (!cfg.autoDownload) {
            std::cerr << "arboocr_demo: --download-models and --no-download contradict"
                      << std::endl;
            return 1;
        }
        const arbo::ocr::ModelPaths paths = arbo::ocr::ensureOcrModels(cfg);
        const std::string labels[4] = {"det ", "cls ", "rec ", "dict"};
        const std::string resolved[4] = {paths.det, paths.cls, paths.rec, paths.dict};
        for (int i = 0; i < 4; ++i) {
            // cls is only fetched with --angle, and the dict is often embedded
            // in the rec ONNX — neither absence is a failure, so say "skipped"
            // rather than "missing" and keep "missing" meaning something wrong.
            const char* state = std::filesystem::exists(resolved[i]) ? "ok     "
                              : (i == 1 && !cfg.useAngleCls)         ? "skipped"
                              : (i == 3)                            ? "absent "
                                                                    : "MISSING";
            std::cout << state << "  " << labels[i] << "  " << resolved[i] << "\n";
        }
        const bool haveRequired = std::filesystem::exists(paths.det)
                               && std::filesystem::exists(paths.rec);
        return haveRequired ? 0 : 2;
    }

    const bool jsonMode = result["json"].as<bool>();

    // Resolve the work list before the models load, so a typo'd list path costs
    // nothing. See readImageList() for why the batch is a file of paths.
    std::vector<std::string> images;
    if (batchMode) {
        const std::string listPath = result["images-from"].as<std::string>();
        if (listPath == "-") {
            images = readImageList(std::cin);
        } else {
            std::ifstream list(listPath, std::ios::binary);
            if (!list) {
                std::cerr << "arboocr_demo: --images-from: cannot open " << listPath << std::endl;
                return 2;
            }
            images = readImageList(list);
        }
        if (images.empty()) {
            std::cerr << "arboocr_demo: --images-from: no image paths in " << listPath << std::endl;
            return 1;
        }
    } else {
        images.push_back(result["image"].as<std::string>());
    }

    // Engine construction loads the ONNX sessions and throws Ort::Exception on
    // a missing/corrupt model. Uncaught, that unwinds past main() the same way
    // a cxxopts parse error would — report the resolved paths and bail cleanly.
    std::unique_ptr<arbo::ocr::Engine> engine;
    try {
        engine = std::make_unique<arbo::ocr::Engine>(cfg);
    } catch (const std::exception& e) {
        const arbo::ocr::ModelPaths paths = arbo::ocr::resolveModelPaths(cfg);
        std::cerr << "arboocr_demo: failed to load models: " << e.what() << "\n"
                  << "  det:  " << paths.det << "\n"
                  << "  cls:  " << paths.cls << "\n"
                  << "  rec:  " << paths.rec << "\n"
                  << "  dict: " << paths.dict << "\n"
                  << "Check --models-dir/--ocr-version/--model-type, or override the"
                     " paths with --det-model/--cls-model/--rec-model/--dict.\n"
                  // Ask the library, not cfg.autoDownload: ARBOOCR_OFFLINE also
                  // suppresses the fetch, and a hint that blames a failed
                  // download sends the user hunting a network fault that never
                  // happened.
                  << (arbo::ocr::modelDownloadsAllowed(cfg)
                          ? "Auto-download was on, so the fetch failed too — rerun with"
                            " --log-level warn to see why.\n"
                          : "Auto-download is off (--no-download / ARBOOCR_OFFLINE).\n")
                  << "Cache dir: " << arbo::ocr::defaultModelsCacheDir() << std::endl;
        return 2;
    }

    if (!jsonMode) {
        std::cout << "Backend: " << engine->backend() << "\n";
    }

    // One Engine, N images. The model load dominates a single page, so amortizing
    // it across the list is the entire point of --images-from; the loop itself is
    // deliberately serial (the Engine already batches crops internally).
    bool anyEmpty = false;
    if (jsonMode && batchMode) std::cout << "[";

    for (size_t imageIdx = 0; imageIdx < images.size(); imageIdx++) {
        const std::string& imagePath = images[imageIdx];

        // recognize() is documented as never throwing; guard anyway so an
        // unexpected inference failure is a diagnostic, not a terminate().
        arbo::ocr::PagePrediction page;
        try {
            page = engine->recognize(imagePath);
        } catch (const std::exception& e) {
            std::cerr << "arboocr_demo: recognition failed: " << e.what() << std::endl;
            return 2;
        }

        // Before the jsonMode early-return, so --draw composes with --json.
        // Diagnostics go to stderr to keep stdout pure JSON in that mode.
        // Single-image only — batch rejected the combination during parsing.
        if (result.count("draw")) {
            const auto& outPath = result["draw"].as<std::string>();
            cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (src.empty()) {
                std::cerr << "arboocr_demo: --draw: cannot re-read image for overlay\n";
            } else if (!cv::imwrite(outPath, arbo::ocr::drawResult(src, page))) {
                std::cerr << "arboocr_demo: --draw: failed to write " << outPath << "\n";
            } else if (!jsonMode) {
                std::cout << "Overlay: " << outPath << "\n";
            }
        }

        // Same placement as --draw, for the same reason: composes with --json, and
        // a write failure is a warning because the recognition itself succeeded.
        if (result.count("markdown")) {
            const auto& outPath = result["markdown"].as<std::string>();
            std::ofstream out(outPath, std::ios::binary);
            if (!out) {
                std::cerr << "arboocr_demo: --markdown: cannot open " << outPath << "\n";
            } else {
                out << arbo::ocr::toMarkdown(page);
                out.close();
                if (!out) {
                    std::cerr << "arboocr_demo: --markdown: failed to write " << outPath << "\n";
                } else if (!jsonMode) {
                    std::cout << "Markdown: " << outPath << "\n";
                }
            }
        }

        if (jsonMode) {
            // --markdown turns word boxes on internally, but the JSON payload is a
            // published contract: "words" appears iff the caller asked for it. Drop
            // them again so --markdown can't silently reshape a wrapper's output.
            if (!result["word-boxes"].as<bool>()) {
                for (auto& line : page.lines) line.words.clear();
            }
            // Pure JSON on stdout, nothing else — callers (e.g. the PHP wrapper)
            // json_decode() the whole stream. Empty lines is still success.
            if (!batchMode) {
                std::cout << arbo::ocr::toJson(page, engine->backend()) << "\n";
                return 0;
            }
            // Batch emits the very same page objects inside a JSON array, so a
            // wrapper json_decode()s the whole stream exactly as it does today.
            // Assembled here with plain brackets rather than a new types.cpp
            // overload, to leave the single-image bare-object shape untouched.
            if (imageIdx) std::cout << ",";
            std::cout << arbo::ocr::toJson(page, engine->backend());
            if (page.lines.empty()) {
                anyEmpty = true;
                std::cerr << "arboocr_demo: no text found in " << imagePath << "\n";
            }
            continue;
        }

        std::cout << "Image: " << page.image << "\n";
        // recognize() never throws — missing/unreadable images and inference
        // failures both yield empty lines with elapsedMs still set. In batch that
        // is one bad page, not a dead run: report it and keep going.
        if (page.lines.empty()) {
            std::cerr << "No text found (missing image, unsupported format, or empty page)"
                      << (batchMode ? ": " + imagePath : std::string())
                      << " (" << page.elapsedMs << " ms)\n";
            if (!batchMode) return 1;
            anyEmpty = true;
            std::cout << "\n";
            continue;
        }
        std::cout << "Lines: " << page.lines.size() << " (" << page.elapsedMs << " ms)\n";
        for (size_t i = 0; i < page.lines.size(); i++) {
            const auto& line = page.lines[i];
            std::cout << "  [" << i << "] \"" << line.text
                      << "\" (score=" << line.score
                      << " detScore=" << line.detScore << ")";
            if (!line.polygon.empty()) {
                std::cout << " poly=[";
                for (size_t p = 0; p < line.polygon.size(); p++) {
                    if (p) std::cout << ", ";
                    std::cout << "(" << line.polygon[p].x << "," << line.polygon[p].y << ")";
                }
                std::cout << "]";
            }
            std::cout << "\n";
        }
        // Blank line between pages keeps the batch scannable; single-image output
        // stays byte-identical to what wrappers already scrape.
        if (batchMode) std::cout << "\n";
    }

    if (jsonMode && batchMode) std::cout << "]\n";
    // 0 = every image had text, 1 = at least one came back empty (partial batch).
    // Single-image paths already returned above, so this only decides the batch.
    return anyEmpty ? 1 : 0;
}
