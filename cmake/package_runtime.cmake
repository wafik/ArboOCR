# Collect the runtime libraries arboocr_demo actually needs into OUT_DIR.
#
# Run in script mode:
#   cmake -DBIN=<path to arboocr_demo[.exe]> -DOUT_DIR=<dir> \
#         -DSEARCH_DIRS=<dir;dir> -P cmake/package_runtime.cmake
#
# Replaces "copy every .dll next to the binary", which shipped eleven OpenCV
# modules (dnn, calib3d, objdetect, stitching, features2d, photo, video,
# videoio, ml, flann, highgui — 12.1 MB on Windows) that nothing ever loads.
# arboOCR includes only opencv2/core, opencv2/imgproc and opencv2/imgcodecs.
#
# A hand-written allowlist would have been shorter and would rot the first
# time vcpkg changed a transitive dependency, failing as a missing DLL at a
# user's first run. GET_RUNTIME_DEPENDENCIES derives the closure instead.

cmake_minimum_required(VERSION 3.21)

# Normalize mixed C:\Windows\system32/foo.dll separators before the exclude
# regexes run, so "[Ss]ystem32" matches reliably.
if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

foreach(required BIN OUT_DIR SEARCH_DIRS)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "package_runtime.cmake: -D${required}=... is required")
    endif()
endforeach()
if(NOT EXISTS "${BIN}")
    message(FATAL_ERROR "package_runtime.cmake: no binary at ${BIN}")
endif()

file(MAKE_DIRECTORY "${OUT_DIR}")

file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${BIN}"
    RESOLVED_DEPENDENCIES_VAR resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    DIRECTORIES ${SEARCH_DIRS}
    # Windows API sets and the UCRT/VC runtime ship with the OS or the
    # redistributable; bundling them is not ours to do.
    PRE_EXCLUDE_REGEXES
        "api-ms-win-.*" "ext-ms-.*"
        "^(kernel32|advapi32|setupapi|dbghelp|dxgi|msvcp[0-9_]+|vcruntime[0-9_]+)\\.dll$"
    # Anything already in a system location is not ours to redistribute.
    POST_EXCLUDE_REGEXES
        "[Ss]ystem32" "[Ww]indows[/\\\\]" "^/lib" "^/usr/lib"
)

set(copied "")
foreach(lib IN LISTS resolved)
    file(COPY "${lib}" DESTINATION "${OUT_DIR}" FOLLOW_SYMLINK_CHAIN)
    list(APPEND copied "${lib}")
endforeach()

# ONNX Runtime dlopen()s its execution providers, so they appear in no import
# table and the resolver above cannot see them. Without these, a package looks
# fine on CPU and then fails to find CUDA/TensorRT at runtime — exactly the
# regression an import-driven allowlist invites. Copy them by name.
foreach(dir IN LISTS SEARCH_DIRS)
    file(GLOB providers
        "${dir}/onnxruntime_providers_*.dll"
        "${dir}/libonnxruntime_providers_*.so")
    foreach(p IN LISTS providers)
        file(COPY "${p}" DESTINATION "${OUT_DIR}")
        list(APPEND copied "${p}")
    endforeach()
endforeach()

if(unresolved)
    # Not fatal: system libraries legitimately resolve outside our search
    # dirs. Printed so a genuinely missing vcpkg library is visible in the
    # build log rather than discovered by a user.
    list(REMOVE_DUPLICATES unresolved)
    message(STATUS "package_runtime: unresolved (expected for system libs):")
    foreach(u IN LISTS unresolved)
        message(STATUS "    ${u}")
    endforeach()
endif()

list(REMOVE_DUPLICATES copied)
list(LENGTH copied count)
set(total 0)
message(STATUS "package_runtime: bundled ${count} runtime libraries")
foreach(lib IN LISTS copied)
    file(SIZE "${lib}" bytes)
    math(EXPR total "${total} + ${bytes}")
    math(EXPR kb "${bytes} / 1024")
    get_filename_component(name "${lib}" NAME)
    message(STATUS "    ${name} (${kb} KB)")
endforeach()
math(EXPR total_mb "${total} / 1048576")
message(STATUS "package_runtime: ${total_mb} MB of runtime libraries")
