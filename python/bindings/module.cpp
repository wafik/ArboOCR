// pybind11 bindings for arboOCR Engine facade (v1).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

#include "arboOCR/engine.hpp"
#include "arboOCR/markdown.hpp"
#include "arboOCR/types.hpp"

namespace py = pybind11;
using namespace arbo::ocr;

namespace {

cv::Mat numpyToBgrMat(const py::array& arr) {
    if (arr.ndim() != 3 || arr.shape(2) != 3) {
        throw std::invalid_argument(
            "image array must be HxWx3 (got ndim=" + std::to_string(arr.ndim()) + ")");
    }
    // Request C-contiguous uint8 view; forcecast only if already integer-like 8-bit.
    if (arr.dtype().kind() != 'u' || arr.itemsize() != 1) {
        throw std::invalid_argument("image array dtype must be uint8 (BGR)");
    }
    auto buf = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!buf) {
        throw std::invalid_argument("could not convert image array to contiguous uint8");
    }
    const int h = static_cast<int>(buf.shape(0));
    const int w = static_cast<int>(buf.shape(1));
    cv::Mat view(h, w, CV_8UC3, const_cast<uint8_t*>(buf.data()));
    return view.clone(); // own memory; Engine may outlive the Python buffer
}

std::string pathLikeToString(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    // pathlib.Path and os.PathLike: str(obj)
    return py::str(obj).cast<std::string>();
}

// bytes/bytearray/memoryview all expose the buffer protocol, which is what a
// Python web handler or queue consumer already holds. Going through a temp
// file to reach recognize(path) would be a syscall round-trip for nothing.
PagePrediction recognizeEncodedBuffer(Engine& self, const py::object& image) {
    py::buffer_info info = py::reinterpret_borrow<py::buffer>(image).request();
    if (info.ndim != 1 || info.itemsize != 1) {
        throw std::invalid_argument("encoded image buffer must be 1-D bytes");
    }
    // buffer_info holds the Py_buffer for its lifetime, so ptr stays valid
    // across this synchronous call.
    return self.recognizeEncoded(static_cast<const uint8_t*>(info.ptr),
                                 static_cast<size_t>(info.size));
}

PagePrediction recognizeDispatch(Engine& self, const py::object& image) {
    // Order matters twice over.
    //
    // Bytes first: py::str(bytes) stringifies to "b'\\x89PNG...'", which would
    // sail into the path branch and be read as a filename.
    //
    // And numpy last: py::isinstance<py::array> imports numpy on first use, so
    // testing it up front made path and bytes input fail on a numpy-free
    // install for no reason. Only the array branch actually needs it.
    if (py::isinstance<py::bytes>(image) || py::isinstance<py::bytearray>(image)
        || py::isinstance<py::memoryview>(image)) {
        return recognizeEncodedBuffer(self, image);
    }
    if (py::isinstance<py::str>(image) || py::hasattr(image, "__fspath__")) {
        return self.recognize(pathLikeToString(image));
    }
    if (py::isinstance<py::array>(image)) {
        cv::Mat mat = numpyToBgrMat(py::array::ensure(image));
        return self.recognize(mat);
    }
    throw std::invalid_argument(
        "image must be a path (str/os.PathLike), encoded bytes/bytearray/"
        "memoryview, or an HxWx3 uint8 BGR numpy array");
}

py::dict modelPathsToDict(const ModelPaths& p) {
    py::dict d;
    d["det"] = p.det;
    d["cls"] = p.cls;
    d["rec"] = p.rec;
    d["dict"] = p.dict;
    return d;
}

} // namespace

PYBIND11_MODULE(_arboocr, m) {
    m.doc() = "arboOCR Python bindings (Engine facade)";

    py::class_<Point2f>(m, "Point2f")
        .def(py::init<>())
        .def(py::init([](float x, float y) { return Point2f{x, y}; }), py::arg("x"), py::arg("y"))
        .def_readwrite("x", &Point2f::x)
        .def_readwrite("y", &Point2f::y)
        .def("__repr__", [](const Point2f& p) {
            return "Point2f(x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y) + ")";
        });

    py::class_<WordBox>(m, "WordBox",
        "One word (one character for CJK) with its own polygon.\n"
        "Polygons come from CTC timestep alignment, so they run roughly half a\n"
        "character wide of true glyph extents — fine for highlighting, not for\n"
        "cropping glyphs.")
        .def(py::init<>())
        .def_readwrite("polygon", &WordBox::polygon)
        .def_readwrite("text", &WordBox::text)
        .def_readwrite("score", &WordBox::score)
        .def("__repr__", [](const WordBox& w) {
            return "WordBox(text='" + w.text + "', score=" + std::to_string(w.score) + ")";
        });

    py::class_<LinePrediction>(m, "LinePrediction")
        .def(py::init<>())
        .def_readwrite("polygon", &LinePrediction::polygon)
        .def_readwrite("text", &LinePrediction::text)
        .def_readwrite("score", &LinePrediction::score)
        .def_readwrite("det_score", &LinePrediction::detScore)
        .def_readwrite("words", &LinePrediction::words,
                       "Per-word polygons; empty unless config.return_word_boxes.");

    py::class_<PagePrediction>(m, "PagePrediction")
        .def(py::init<>())
        .def_readwrite("image", &PagePrediction::image)
        .def_readwrite("lines", &PagePrediction::lines)
        .def_readwrite("elapsed_ms", &PagePrediction::elapsedMs);

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("ocr_version", &EngineConfig::ocrVersion)
        .def_readwrite("model_type", &EngineConfig::modelType)
        .def_readwrite("det_box_thresh", &EngineConfig::detBoxThresh)
        .def_readwrite("det_thresh", &EngineConfig::detThresh)
        .def_readwrite("det_unclip_ratio", &EngineConfig::detUnclipRatio)
        .def_readwrite("det_limit_side_len", &EngineConfig::detLimitSideLen)
        .def_readwrite("rec_batch_num", &EngineConfig::recBatchNum)
        .def_readwrite("use_angle_cls", &EngineConfig::useAngleCls)
        .def_readwrite("use_cuda", &EngineConfig::useCuda)
        .def_readwrite("use_tensorrt", &EngineConfig::useTensorrt)
        .def_readwrite("use_fp16", &EngineConfig::useFp16)
        .def_readwrite("intra_op_num_threads", &EngineConfig::intraOpNumThreads)
        .def_readwrite("inter_op_num_threads", &EngineConfig::interOpNumThreads)
        .def_readwrite("use_clahe", &EngineConfig::useClahe)
        .def_readwrite("split_overmerged", &EngineConfig::splitOvermerged)
        .def_readwrite("minimum_confidence", &EngineConfig::minimumConfidence)
        .def_readwrite("return_word_boxes", &EngineConfig::returnWordBoxes)
        .def_readwrite("trt_cache_dir", &EngineConfig::trtCacheDir)
        .def_readwrite("models_dir", &EngineConfig::modelsDir)
        .def_readwrite("det_model_path", &EngineConfig::detModelPath)
        .def_readwrite("cls_model_path", &EngineConfig::clsModelPath)
        .def_readwrite("rec_model_path", &EngineConfig::recModelPath)
        .def_readwrite("dict_path", &EngineConfig::dictPath);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const EngineConfig&>(), py::arg("config"))
        .def("backend", &Engine::backend)
        .def("recognize", &recognizeDispatch, py::arg("image"),
             "Run OCR on an image path (str/Path), encoded bytes/bytearray/\n"
             "memoryview, or an HxWx3 uint8 BGR numpy array.\n"
             "Never raises for missing/unreadable/undecodable images — returns\n"
             "empty lines with elapsed_ms still set.")
        .def("recognize_encoded", &recognizeEncodedBuffer, py::arg("data"),
             "Run OCR on encoded image bytes (PNG/JPEG/...). Explicit form of\n"
             "recognize() for callers that want the intent in the call site.");

    m.def("to_json",
          [](const PagePrediction& page, bool pretty) { return toJson(page, pretty); },
          py::arg("page"), py::arg("pretty") = false);

    m.def("to_json",
          [](const LinePrediction& line, bool pretty) { return toJson(line, pretty); },
          py::arg("line"), py::arg("pretty") = false);

    m.def("to_markdown",
          [](const PagePrediction& page) { return toMarkdown(page); },
          py::arg("page"),
          "Reconstruct a rough markdown document from a recognized page.\n"
          "Structure is inferred from line geometry: paragraphs, headings,\n"
          "list items, and `key | value` table rows (needs return_word_boxes).");

    m.def("resolve_model_paths",
          [](const EngineConfig& cfg) { return modelPathsToDict(resolveModelPaths(cfg)); },
          py::arg("config"));

    m.def("detect_cuda", &detectCuda);
    m.def("detect_tensorrt", &detectTensorrt);
}
