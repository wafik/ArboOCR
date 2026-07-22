#pragma once
// Internal helper shared by Detector/Classifier/Recognizer's loadModel().
// Not part of arboOCR's public API — lives under src/, not include/.

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace arbo::ocr::detail {

/// Every input tensor name for `session`, keeping the allocator-owned
/// strings alive via the returned AllocatedStringPtr vector.
std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session* session);

/// Every output tensor name for `session` (same ownership semantics).
std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session* session);

/// Literal TensorRT profile shape strings, e.g. "x:1x3x32x32" — exactly
/// what ONNXRuntime's trt_profile_*_shapes options expect (tensor name +
/// dims). Detector/Classifier/Recognizer each pin a different profile
/// because their input tensors have different natural size ranges
/// (full-page image vs. fixed 192x48 crop vs. variable-width text-line
/// crop).
struct TrtShapeProfile {
    std::string minShape;
    std::string optShape;
    std::string maxShape;
};

/// Appends TensorRT (if useTensorrt) then CUDA (if useCuda) execution
/// providers to `sessionOptions`, in that order — ONNXRuntime tries
/// providers in append order, so TensorRT must come first. Shared by all
/// three net classes; only the profile shape, cache dir, and FP16 flag
/// differ between callers. `useFp16` maps to trt_fp16_enable (default on
/// for edge latency; ignored when TensorRT is off).
void configureExecutionProviders(
    Ort::SessionOptions& sessionOptions,
    bool useCuda,
    bool useTensorrt,
    const std::string& trtCacheDir,
    const TrtShapeProfile& profile,
    bool useFp16 = true
);

} // namespace arbo::ocr::detail
