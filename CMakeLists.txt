cmake_minimum_required(VERSION 3.20)
project(arboOCR VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# Default (vcpkg): onnxruntime/OpenCV/CURL/doctest/cxxopts via CONFIG packages.
# System-deps (-DARBOOCR_USE_SYSTEM_DEPS=ON): apt OpenCV/CURL + vendored
# onnxruntime headers, runtime lib from ARBOOCR_ORT_LIB_DIR (Jetson/aarch64
# with a CUDA/TensorRT-enabled onnxruntime).
# Prebuilt 1.28.0 (-DARBOOCR_ORT_VERSION=1.28.0): official Microsoft release
# archives instead of the vcpkg 1.23.2 port — see cmake/ort_prebuilt.cmake.
# Windows/Linux x64 only; cannot combine with ARBOOCR_USE_SYSTEM_DEPS.
option(ARBOOCR_USE_SYSTEM_DEPS "Use apt packages + vendored onnxruntime (Jetson/aarch64)" OFF)
set(ARBOOCR_ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime/dist/include"
    CACHE PATH "onnxruntime C++ headers (system-deps mode)")
set(ARBOOCR_ORT_LIB_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime/lib"
    CACHE PATH "Directory containing libonnxruntime.so (system-deps mode) — defaults to the vendored copy")

# onnxruntime source: 1.23.2 = vcpkg port (default, CI stays green), 1.28.0 =
# official prebuilt archives (opt-in — no vcpkg port exists past 1.23.2).
set(ARBOOCR_ORT_VERSION "1.23.2" CACHE STRING "onnxruntime version: 1.23.2 (vcpkg) or 1.28.0 (official prebuilt)")
set_property(CACHE ARBOOCR_ORT_VERSION PROPERTY STRINGS "1.23.2" "1.28.0")
if(NOT ARBOOCR_ORT_VERSION STREQUAL "1.23.2" AND NOT ARBOOCR_ORT_VERSION STREQUAL "1.28.0")
    message(FATAL_ERROR "ARBOOCR_ORT_VERSION=${ARBOOCR_ORT_VERSION} unsupported (want 1.23.2 or 1.28.0)")
endif()
set(ARBOOCR_ORT_PREBUILT OFF CACHE INTERNAL "ON when prebuilt ORT 1.28.0 is active")
if(ARBOOCR_ORT_VERSION STREQUAL "1.28.0")
    if(ARBOOCR_USE_SYSTEM_DEPS)
        message(FATAL_ERROR "ARBOOCR_ORT_VERSION=1.28.0 cannot combine with ARBOOCR_USE_SYSTEM_DEPS=ON "
            "(the prebuilt path replaces vcpkg ORT; Jetson keeps its own vendored flow)")
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ort_prebuilt.cmake)
    # Reuse the system-deps wiring below (plain `onnxruntime` link + link dir +
    # RPATH) with the prebuilt paths, so link/packaging/install need no
    # further special cases. vcpkg still installs the unused 1.23.2 port
    # (vcpkg.json is unchanged); the build just never finds or links it.
    set(ARBOOCR_ORT_INCLUDE_DIR "${ARBOOCR_ORT_PREBUILT_ROOT}/include"
        CACHE PATH "onnxruntime C++ headers (prebuilt 1.28.0 mode)" FORCE)
    set(ARBOOCR_ORT_LIB_DIR "${ARBOOCR_ORT_PREBUILT_ROOT}/lib"
        CACHE PATH "Directory containing the prebuilt onnxruntime 1.28.0 runtime" FORCE)
    set(ARBOOCR_ORT_PREBUILT ON CACHE INTERNAL "ON when prebuilt ORT 1.28.0 is active" FORCE)
endif()

find_package(doctest CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

if(ARBOOCR_USE_SYSTEM_DEPS OR ARBOOCR_ORT_PREBUILT)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    if(ARBOOCR_ORT_PREBUILT)
        # Prebuilt 1.28.0: only onnxruntime leaves the vcpkg path; OpenCV and
        # CURL still come from the vcpkg CONFIG packages.
        find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
        find_package(CURL CONFIG REQUIRED)
    else()
        find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
        find_package(CURL REQUIRED)
    endif()
    if(ARBOOCR_ORT_LIB_DIR STREQUAL "" OR NOT EXISTS "${ARBOOCR_ORT_LIB_DIR}")
        message(FATAL_ERROR "No onnxruntime runtime in ${ARBOOCR_ORT_LIB_DIR} "
            "(system-deps: vendor it there, or pass -DARBOOCR_ORT_LIB_DIR=<dir> to point elsewhere — see README)")
    endif()
else()
    find_package(onnxruntime CONFIG REQUIRED)
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
    find_package(CURL CONFIG REQUIRED)
endif()

# Vendored Clipper (polygon offset for unClipBox). Exported alongside arboOCR
# because a static library that links another static library leaves those
# symbols unresolved for the consumer; clipper.hpp itself stays private
# (no public arboOCR header includes it), so it gets no INSTALL_INTERFACE.
add_library(arboocr_clipper STATIC vendor/rapidocronnx/clipper.cpp)
target_include_directories(arboocr_clipper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/rapidocronnx>)

# Core library.
add_library(arboOCR STATIC
    src/arboOCR/ort_provider_utils.cpp
    src/arboOCR/ocr_utils.cpp
    src/arboOCR/preprocess.cpp
    src/arboOCR/detector.cpp
    src/arboOCR/classifier.cpp
    src/arboOCR/recognizer.cpp
    src/arboOCR/engine.cpp
    src/arboOCR/visualize.cpp
    src/arboOCR/markdown.cpp
    src/arboOCR/model_downloader.cpp
    src/arboOCR/types.cpp
    src/arboOCR/logging.cpp
)
target_include_directories(arboOCR PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    "$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>")
# OpenCV_LIBS is now exactly the three components requested above, not every
# module OpenCV happens to have built. vcpkg.json turns the rest off at the
# source: arboOCR includes only opencv2/core, opencv2/imgproc and
# opencv2/imgcodecs, so dnn/calib3d/highgui/gapi/... were build time nobody
# spent usefully. Adding a new opencv2/<module> include means adding the
# component here and the feature in vcpkg.json — the build fails first.
target_link_libraries(arboOCR PUBLIC ${OpenCV_LIBS} CURL::libcurl arboocr_clipper)
add_library(arboOCR::arboOCR ALIAS arboOCR)

if(ARBOOCR_USE_SYSTEM_DEPS OR ARBOOCR_ORT_PREBUILT)
    # Vendored/prebuilt ORT paths live outside any CMake package, so they
    # cannot be exported verbatim; arboOCRConfig.cmake re-applies them for
    # installed consumers.
    #
    # Both archives lay headers flat (include/onnxruntime_cxx_api.h — no
    # onnxruntime/ subdir like the vcpkg port), matching what the code
    # includes, so no extra include shim is needed here.
    target_include_directories(arboOCR PUBLIC $<BUILD_INTERFACE:${ARBOOCR_ORT_INCLUDE_DIR}>)
    target_link_directories(arboOCR PUBLIC $<BUILD_INTERFACE:${ARBOOCR_ORT_LIB_DIR}>)
    target_link_libraries(arboOCR PUBLIC onnxruntime)
    set_target_properties(arboOCR PROPERTIES
        INSTALL_RPATH "${ARBOOCR_ORT_LIB_DIR}" BUILD_WITH_INSTALL_RPATH TRUE)
else()
    # Konfigurasi bawaan yang sudah berjalan baik di Windows via vcpkg
    target_link_libraries(arboOCR PUBLIC onnxruntime::onnxruntime)
    
    # Tambahan spesifik untuk Linux agar tidak mengganggu build Windows
    if(UNIX AND NOT APPLE)
        # Opsi 1: Jika vcpkg menaruh header di path spesifik yang tidak ter-link otomatis
        # find_path() akan mencari lokasi onnxruntime_cxx_api.h dan kita menambahkannya secara manual
        find_path(ONNXRUNTIME_INCLUDE_DIR 
            NAMES onnxruntime_cxx_api.h 
            PATH_SUFFIXES onnxruntime/core/session onnxruntime
        )
        if(ONNXRUNTIME_INCLUDE_DIR)
            # BUILD_INTERFACE: under vcpkg this resolves inside the build tree
            # (vcpkg_installed/), and install(EXPORT) hard-errors on interface
            # paths there. Installed consumers get the header dir from
            # find_dependency(onnxruntime) in arboOCRConfig.cmake instead.
            target_include_directories(arboOCR PUBLIC
                $<BUILD_INTERFACE:${ONNXRUNTIME_INCLUDE_DIR}>)
        endif()
    endif()
endif()

# Tests.
enable_testing()
add_executable(arboocr_tests
    tests/test_main.cpp
    tests/test_ocr_utils.cpp
    tests/test_preprocess.cpp
    tests/test_detector.cpp
    tests/test_classifier.cpp
    tests/test_recognizer.cpp
    tests/test_engine.cpp
    tests/test_engine_inference.cpp
    tests/test_visualize.cpp
    tests/test_markdown.cpp
    tests/test_model_downloader.cpp
    tests/test_logging.cpp
)
target_link_libraries(arboocr_tests PRIVATE arboOCR doctest::doctest)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(arboocr_tests PRIVATE stdc++fs)
endif()
if(ARBOOCR_USE_SYSTEM_DEPS OR ARBOOCR_ORT_PREBUILT)
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
if(ARBOOCR_USE_SYSTEM_DEPS OR ARBOOCR_ORT_PREBUILT)
    set_target_properties(arboocr_demo PROPERTIES
        INSTALL_RPATH "${ARBOOCR_ORT_LIB_DIR}" BUILD_WITH_INSTALL_RPATH TRUE)
endif()
if(ARBOOCR_ORT_PREBUILT)
    # Copy the prebuilt ORT runtime next to every executable that loads it
    # (demo + tests so ctest runs) — then package_runtime.cmake resolves it
    # with the stock SEARCH_DIRS (binary-dir glob for
    # onnxruntime_providers_*; GET_RUNTIME for onnxruntime.dll/.so via the
    # import lib / link dir).
    #
    # Linux note: the copy is belt-and-braces — BUILD_WITH_INSTALL_RPATH on
    # the prebuilt lib dir would already resolve it. But the explicit copy
    # makes the tree relocatable and matches the Windows behavior.
    foreach(_ort_exe arboocr_demo arboocr_tests)
        add_custom_command(TARGET ${_ort_exe} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ARBOOCR_ORT_PREBUILT_RUNTIME_FILES}
                $<TARGET_FILE_DIR:${_ort_exe}>)
    endforeach()
endif()

# Install + package export, so consumers can find_package(arboOCR CONFIG) and
# link arboOCR::arboOCR instead of vendoring the tree. On by default only for a
# standalone build (3.20 predates PROJECT_IS_TOP_LEVEL, hence the manual check).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ARBOOCR_INSTALL_DEFAULT ON)
else()
    set(ARBOOCR_INSTALL_DEFAULT OFF)
endif()
option(ARBOOCR_INSTALL "Generate install rules and the arboOCR CMake package" ${ARBOOCR_INSTALL_DEFAULT})

if(ARBOOCR_INSTALL)
    include(CMakePackageConfigHelpers)
    set(ARBOOCR_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/arboOCR")

    install(TARGETS arboOCR arboocr_clipper
        EXPORT arboOCRTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # The Go/Rust/PHP wrappers shell out to this binary. Tests and examples
    # stay build-tree only.
    install(TARGETS arboocr_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT arboOCRTargets
        FILE arboOCRTargets.cmake
        NAMESPACE arboOCR::
        DESTINATION ${ARBOOCR_CMAKEDIR})

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/arboOCRConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/arboOCRConfig.cmake
        INSTALL_DESTINATION ${ARBOOCR_CMAKEDIR})
    # Pre-1.0: 0.1 and 0.2 are not interchangeable, so match on the minor too.
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/arboOCRConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/arboOCRConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/arboOCRConfigVersion.cmake
        DESTINATION ${ARBOOCR_CMAKEDIR})
endif()

# Buildable usage examples (see examples/README.md).
add_subdirectory(examples)

# Optional Python bindings (pybind11). Off by default so C++-only builds
# never require Python/pybind11. See python/ and README "Python" section.
option(ARBOOCR_BUILD_PYTHON "Build pybind11 module arboocr._arboocr" OFF)
if(ARBOOCR_BUILD_PYTHON)
    add_subdirectory(python)
endif()
