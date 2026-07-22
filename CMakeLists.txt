cmake_minimum_required(VERSION 3.20)
project(arboOCR CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default (vcpkg): onnxruntime/OpenCV/CURL/doctest/cxxopts via CONFIG packages.
# System-deps (-DARBOOCR_USE_SYSTEM_DEPS=ON): apt OpenCV/CURL + vendored
# onnxruntime headers, runtime lib from ARBOOCR_ORT_LIB_DIR (Jetson/aarch64
# with a CUDA/TensorRT-enabled onnxruntime).
option(ARBOOCR_USE_SYSTEM_DEPS "Use apt packages + vendored onnxruntime (Jetson/aarch64)" OFF)
set(ARBOOCR_ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime/dist/include"
    CACHE PATH "onnxruntime C++ headers (system-deps mode)")
set(ARBOOCR_ORT_LIB_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime/lib"
    CACHE PATH "Directory containing libonnxruntime.so (system-deps mode) — defaults to the vendored copy")

find_package(doctest CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

if(ARBOOCR_USE_SYSTEM_DEPS)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    find_package(OpenCV REQUIRED)
    find_package(CURL REQUIRED)
    if(ARBOOCR_ORT_LIB_DIR STREQUAL "" OR NOT EXISTS "${ARBOOCR_ORT_LIB_DIR}")
        message(FATAL_ERROR "ARBOOCR_USE_SYSTEM_DEPS=ON needs libonnxruntime.so in ${ARBOOCR_ORT_LIB_DIR} "
            "(vendor it there, or pass -DARBOOCR_ORT_LIB_DIR=<dir> to point elsewhere — see README)")
    endif()
else()
    find_package(onnxruntime CONFIG REQUIRED)
    find_package(OpenCV CONFIG REQUIRED)
    find_package(CURL CONFIG REQUIRED)
endif()

# Vendored Clipper (polygon offset for unClipBox).
add_library(arboocr_clipper STATIC vendor/rapidocronnx/clipper.cpp)
target_include_directories(arboocr_clipper PUBLIC vendor/rapidocronnx)

# Core library.
add_library(arboOCR STATIC
    src/arboOCR/ort_provider_utils.cpp
    src/arboOCR/ocr_utils.cpp
    src/arboOCR/detector.cpp
    src/arboOCR/classifier.cpp
    src/arboOCR/recognizer.cpp
    src/arboOCR/engine.cpp
    src/arboOCR/model_downloader.cpp
    src/arboOCR/types.cpp
    src/arboOCR/logging.cpp
)
target_include_directories(arboOCR PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(arboOCR PUBLIC ${OpenCV_LIBS} CURL::libcurl arboocr_clipper)

if(ARBOOCR_USE_SYSTEM_DEPS)
    target_include_directories(arboOCR PUBLIC ${ARBOOCR_ORT_INCLUDE_DIR})
    target_link_directories(arboOCR PUBLIC ${ARBOOCR_ORT_LIB_DIR})
    target_link_libraries(arboOCR PUBLIC onnxruntime)
    set_target_properties(arboOCR PROPERTIES
        INSTALL_RPATH "${ARBOOCR_ORT_LIB_DIR}" BUILD_WITH_INSTALL_RPATH TRUE)
else()
    target_link_libraries(arboOCR PUBLIC onnxruntime::onnxruntime)
endif()

# Tests.
enable_testing()
add_executable(arboocr_tests
    tests/test_main.cpp
    tests/test_ocr_utils.cpp
    tests/test_detector.cpp
    tests/test_classifier.cpp
    tests/test_recognizer.cpp
    tests/test_engine.cpp
    tests/test_engine_inference.cpp
    tests/test_model_downloader.cpp
    tests/test_logging.cpp
)
target_link_libraries(arboocr_tests PRIVATE arboOCR doctest::doctest)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(arboocr_tests PRIVATE stdc++fs)
endif()
if(ARBOOCR_USE_SYSTEM_DEPS)
    set_target_properties(arboocr_tests PROPERTIES
        INSTALL_RPATH "${ARBOOCR_ORT_LIB_DIR}" BUILD_WITH_INSTALL_RPATH TRUE)
endif()
add_test(NAME arboocr_tests COMMAND arboocr_tests
    WORKING_DIRECTORY $<TARGET_FILE_DIR:arboocr_tests>)

# Copy fixtures (and models/ if present) next to the test binary so relative
# paths resolve. models/ is gitignored; the copy is best-effort.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
add_custom_command(TARGET arboocr_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/tests/fixtures
        $<TARGET_FILE_DIR:arboocr_tests>/tests/fixtures
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:arboocr_tests>/models
)

# Demo CLI.
add_executable(arboocr_demo cli/arboocr_demo.cpp)
target_link_libraries(arboocr_demo PRIVATE arboOCR cxxopts::cxxopts)
if(ARBOOCR_USE_SYSTEM_DEPS)
    set_target_properties(arboocr_demo PROPERTIES
        INSTALL_RPATH "${ARBOOCR_ORT_LIB_DIR}" BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Buildable usage examples (see examples/README.md).
add_subdirectory(examples)
