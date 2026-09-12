# Official prebuilt ONNXRuntime 1.28.0 for Windows/Linux x64.
# Activated by -DARBOOCR_ORT_VERSION=1.28.0 (default 1.23.2 keeps vcpkg CI green).
#
# vcpkg has no 1.28 port (local registry and upstream max out at 1.23.2), so
# this path fetches Microsoft's official release archives instead of bumping
# a vcpkg version that does not exist. Pinned per-OS archives (single source
# of truth for URLs + hashes — NEVER invent download URLs, extend this table):
#
#   Windows x64: onnxruntime-win-x64-1.28.0.zip
#     https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-win-x64-1.28.0.zip
#     SHA256 ABEF733DACBE2F571547A7150B479B5CB9CC0DF22F96C24983A42CADB1B4F8BC
#   Linux x64:   onnxruntime-linux-x64-1.28.0.tgz
#     https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz
#     SHA256 A3E1B79D7BB1BF09696CE675F49E4064E6C81F6202B8225624FFF0E93F8D6407
#
# A DLL-swap experiment already proved the 1.28.0 runtime safe with this
# codebase (1.23.2 headers + 1.28.0 DLLs: 6/6 byte-identical outputs), so no
# source changes are needed — only build/packaging wiring, which is what this
# module does.
#
# Layout: the prebuilt tree's include/ holds the C++ headers FLAT
# (include/onnxruntime_cxx_api.h — NOT include/onnxruntime/ like the vcpkg
# port), and lib/ holds the runtime + import lib. The build links the plain
# `onnxruntime` library name against ARBOOCR_ORT_LIB_DIR (same wiring as the
# Jetson system-deps flow), so no imported-target shim is needed and the
# release.yml SEARCH_DIRS globs need no changes.
#
# When active, vcpkg still installs its own onnxruntime 1.23.2 (manifest mode
# installs every vcpkg.json dependency unconditionally) but nothing references
# it: the build just never calls find_package(onnxruntime).

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ort-prebuilt: ARBOOCR_ORT_VERSION=1.28.0 supports Windows/Linux x64 only "
        "(aarch64 keeps -DARBOOCR_USE_SYSTEM_DEPS=ON — see README's Jetson section)")
endif()

set(ARBOOCR_ORT_PREBUILT_VERSION "1.28.0")
set(_ORT_PREBUILT_BASE_URL
    "https://github.com/microsoft/onnxruntime/releases/download/v${ARBOOCR_ORT_PREBUILT_VERSION}")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_ORT_ARCHIVE_NAME "onnxruntime-win-x64-${ARBOOCR_ORT_PREBUILT_VERSION}.zip")
    set(_ORT_ARCHIVE_SHA256
        "ABEF733DACBE2F571547A7150B479B5CB9CC0DF22F96C24983A42CADB1B4F8BC")
    set(_ORT_ROOT_DIR_NAME "onnxruntime-win-x64-${ARBOOCR_ORT_PREBUILT_VERSION}")
else()
    set(_ORT_ARCHIVE_NAME "onnxruntime-linux-x64-${ARBOOCR_ORT_PREBUILT_VERSION}.tgz")
    set(_ORT_ARCHIVE_SHA256
        "A3E1B79D7BB1BF09696CE675F49E4064E6C81F6202B8225624FFF0E93F8D6407")
    set(_ORT_ROOT_DIR_NAME "onnxruntime-linux-x64-${ARBOOCR_ORT_PREBUILT_VERSION}")
endif()

# Offline escape hatch: point at a local copy instead of downloading.
# The SHA256 check below still applies, so a stale/wrong file fails loudly.
set(ARBOOCR_ORT_ARCHIVE_FILE "" CACHE FILEPATH
    "Local prebuilt onnxruntime archive (skips the download; hash still verified)")

set(_ORT_DL_DIR "${CMAKE_BINARY_DIR}/_deps/ort-prebuilt")
set(_ORT_ARCHIVE "${_ORT_DL_DIR}/${_ORT_ARCHIVE_NAME}")
file(MAKE_DIRECTORY "${_ORT_DL_DIR}")

if(ARBOOCR_ORT_ARCHIVE_FILE)
    if(NOT EXISTS "${ARBOOCR_ORT_ARCHIVE_FILE}")
        message(FATAL_ERROR
            "ort-prebuilt: ARBOOCR_ORT_ARCHIVE_FILE points nowhere: ${ARBOOCR_ORT_ARCHIVE_FILE}")
    endif()
    get_filename_component(_ORT_GIVEN_NAME "${ARBOOCR_ORT_ARCHIVE_FILE}" NAME)
    if(NOT _ORT_GIVEN_NAME STREQUAL _ORT_ARCHIVE_NAME)
        message(FATAL_ERROR "ort-prebuilt: ARBOOCR_ORT_ARCHIVE_FILE must be named "
            "${_ORT_ARCHIVE_NAME} (got ${_ORT_GIVEN_NAME}) — prevents mixing up the OS archives")
    endif()
    file(COPY "${ARBOOCR_ORT_ARCHIVE_FILE}" DESTINATION "${_ORT_DL_DIR}")
endif()

if(NOT EXISTS "${_ORT_ARCHIVE}")
    message(STATUS "ort-prebuilt: downloading ${_ORT_PREBUILT_BASE_URL}/${_ORT_ARCHIVE_NAME}")
    file(DOWNLOAD "${_ORT_PREBUILT_BASE_URL}/${_ORT_ARCHIVE_NAME}" "${_ORT_ARCHIVE}"
        EXPECTED_HASH SHA256=${_ORT_ARCHIVE_SHA256}
        SHOW_PROGRESS
        TLS_VERIFY ON)
else()
    # file(DOWNLOAD) only verifies EXPECTED_HASH on a fresh download, so check
    # explicitly: covers the ARBOOCR_ORT_ARCHIVE_FILE copy and any
    # stale/partial file left behind by an interrupted configure.
    file(SHA256 "${_ORT_ARCHIVE}" _ORT_ACTUAL_SHA256)
    string(TOLOWER "${_ORT_ACTUAL_SHA256}" _ORT_ACTUAL_SHA256)
    string(TOLOWER "${_ORT_ARCHIVE_SHA256}" _ORT_EXPECTED_SHA256)
    if(NOT _ORT_ACTUAL_SHA256 STREQUAL _ORT_EXPECTED_SHA256)
        message(FATAL_ERROR "ort-prebuilt: SHA256 mismatch for ${_ORT_ARCHIVE}\n"
            "  expected ${_ORT_ARCHIVE_SHA256}\n"
            "  actual   ${_ORT_ACTUAL_SHA256}\n"
            "Delete the file and reconfigure.")
    endif()
endif()

set(ARBOOCR_ORT_PREBUILT_ROOT "${_ORT_DL_DIR}/${_ORT_ROOT_DIR_NAME}"
    CACHE PATH "Extracted prebuilt onnxruntime 1.28.0 root" FORCE)
if(NOT EXISTS "${ARBOOCR_ORT_PREBUILT_ROOT}/include/onnxruntime_c_api.h")
    message(STATUS "ort-prebuilt: extracting ${_ORT_ARCHIVE_NAME}")
    file(ARCHIVE_EXTRACT INPUT "${_ORT_ARCHIVE}" DESTINATION "${_ORT_DL_DIR}")
endif()
if(NOT EXISTS "${ARBOOCR_ORT_PREBUILT_ROOT}/include/onnxruntime_c_api.h")
    message(FATAL_ERROR "ort-prebuilt: extraction did not produce "
        "${ARBOOCR_ORT_PREBUILT_ROOT}/include/onnxruntime_c_api.h")
endif()

# Header guard: the pinned archives must expose ORT_API_VERSION 28 (vcpkg's
# 1.23.2 exposes 23). Catches URL drift or a misplaced override file before
# any compile error can confuse the picture.
file(STRINGS "${ARBOOCR_ORT_PREBUILT_ROOT}/include/onnxruntime_c_api.h" _ORT_API_LINE
    REGEX "#define ORT_API_VERSION [0-9]+")
if(NOT _ORT_API_LINE MATCHES "ORT_API_VERSION 28")
    message(FATAL_ERROR
        "ort-prebuilt: expected ORT_API_VERSION 28 in prebuilt headers, got: ${_ORT_API_LINE}")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ARBOOCR_ORT_PREBUILT_RUNTIME_FILES
        "${ARBOOCR_ORT_PREBUILT_ROOT}/lib/onnxruntime.dll"
        "${ARBOOCR_ORT_PREBUILT_ROOT}/lib/onnxruntime_providers_shared.dll")
    foreach(_ort_dll IN LISTS ARBOOCR_ORT_PREBUILT_RUNTIME_FILES)
        if(NOT EXISTS "${_ort_dll}")
            message(FATAL_ERROR "ort-prebuilt: missing ${_ort_dll}")
        endif()
    endforeach()
else()
    # NOTE: the tarball's shipped CMake config is unusable — its
    # onnxruntimeTargets-release.cmake references
    # <root>/lib64/libonnxruntime.so.1.28.0, but the archive ships lib/ only
    # (no lib64/). Reason #2 the build links plain `onnxruntime` against
    # ARBOOCR_ORT_LIB_DIR instead of consuming that config.
    set(ARBOOCR_ORT_PREBUILT_RUNTIME_FILES
        "${ARBOOCR_ORT_PREBUILT_ROOT}/lib/libonnxruntime.so.1.28.0"
        "${ARBOOCR_ORT_PREBUILT_ROOT}/lib/libonnxruntime_providers_shared.so")
    foreach(_ort_so IN LISTS ARBOOCR_ORT_PREBUILT_RUNTIME_FILES)
        if(NOT EXISTS "${_ort_so}")
            message(FATAL_ERROR "ort-prebuilt: missing ${_ort_so}")
        endif()
    endforeach()
endif()
message(STATUS "ort-prebuilt: onnxruntime ${ARBOOCR_ORT_PREBUILT_VERSION} ready at ${ARBOOCR_ORT_PREBUILT_ROOT} (ORT_API_VERSION 28)")
