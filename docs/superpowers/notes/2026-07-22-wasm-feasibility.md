# WebAssembly (WASM) support — feasibility note (low priority)

**Date:** 2026-07-22  
**Status:** Parked — not scheduled  
**Priority:** Below path overrides, native edge performance, Python bindings

## Summary

Compiling arboOCR for the browser with Emscripten is **technically possible** but **not a small port**. The C++ core is relatively clean; the hard parts are OpenCV + ONNXRuntime WASM packaging, model size/RAM, and dropping native-only features (CUDA/TensorRT, libcurl).

Do **not** prioritize this until native library APIs and packaging are stable.

## What already helps

- C++17 inference pipeline (detect → classify → recognize)
- `Engine::recognize(const cv::Mat&)` (in-memory input)
- `toJson(PagePrediction)` for structured JS-friendly output
- Optional logging callback (no forced stdout)

## Blockers / costs

| Area | Issue |
|------|--------|
| ONNXRuntime | Need ORT WASM build (CPU; no CUDA/TensorRT) |
| OpenCV | `imgproc` + `imgcodecs` must be WASM-built or replaced with a slim subset |
| libcurl | `model_downloader` is irrelevant in-browser — exclude from WASM target |
| Models | `medium` rec alone ~tens of MB; cold start + tab memory |
| Paths | `std::filesystem` model load → need virtual FS or buffer-based load |
| Async | `std::async` is awkward; Web Workers are the natural model |
| Perf | Browser ≈ CPU WASM; loses Jetson TensorRT/FP16 edge story |

## If revisited later (v0 sketch only)

Non-goals for a first PoC: downloader, CUDA/TRT, full OpenCV highgui, medium/large models.

Possible shape:

1. CMake option e.g. `ARBOOCR_WASM=ON` (Emscripten toolchain)
2. Link ORT WASM + OpenCV core/imgproc (minimal modules)
3. Exclude `model_downloader.cpp` from the WASM target
4. Prefer **tiny** models; load weights from JS `ArrayBuffer` / IDBFS
5. Expose a thin C API or Embind surface: `recognize(imageBytes) → JSON`
6. Run heavy work in a Worker; main thread only posts results

Alternative that may be cheaper product-wise: **ONNX Runtime Web + JS/TS pipeline** instead of full Emscripten of this repo.

## Decision

Documented for awareness only. **No implementation plan until explicitly prioritized.**
