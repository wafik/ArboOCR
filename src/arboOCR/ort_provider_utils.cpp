#include "ort_provider_utils.hpp"

#include <unordered_map>

namespace arbo::ocr::detail {

std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session* session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> names;
    size_t n = session->GetInputCount();
    names.reserve(n);
    for (size_t i = 0; i < n; i++) {
        names.push_back(session->GetInputNameAllocated(i, allocator));
    }
    return names;
}

std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session* session) {
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> names;
    size_t n = session->GetOutputCount();
    names.reserve(n);
    for (size_t i = 0; i < n; i++) {
        names.push_back(session->GetOutputNameAllocated(i, allocator));
    }
    return names;
}

void configureExecutionProviders(
    Ort::SessionOptions& sessionOptions,
    bool useCuda,
    bool useTensorrt,
    const std::string& trtCacheDir,
    const TrtShapeProfile& profile,
    bool useFp16
) {
    // TensorRT must be appended BEFORE CUDA (ORT tries providers in order).
    if (useTensorrt) {
        Ort::TensorRTProviderOptions trtOpts;
        std::unordered_map<std::string, std::string> opts = {
            {"device_id", "0"},
            {"trt_max_workspace_size", "1073741824"}, // 1GB
            {"trt_fp16_enable", useFp16 ? "1" : "0"},
            {"trt_engine_cache_enable", "1"},
            {"trt_profile_min_shapes", profile.minShape},
            {"trt_profile_opt_shapes", profile.optShape},
            {"trt_profile_max_shapes", profile.maxShape},
        };
        if (!trtCacheDir.empty()) {
            opts["trt_engine_cache_path"] = trtCacheDir;
        }
        trtOpts.Update(opts);
        sessionOptions.AppendExecutionProvider_TensorRT_V2(*trtOpts);
    }

    if (useCuda) {
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = 0;
        cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    }
}

} // namespace arbo::ocr::detail
