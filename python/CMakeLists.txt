# Optional Python bindings (pybind11). Enabled from root via ARBOOCR_BUILD_PYTHON=ON.

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Prefer installed pybind11; otherwise fetch a pinned release so C++-only
# vcpkg consumers never pull Python deps when the option is OFF.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_arboocr MODULE bindings/module.cpp)
target_link_libraries(_arboocr PRIVATE arboOCR)
target_include_directories(_arboocr PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Drop the extension next to the pure-Python package for importable layout.
set(_ARBOOCR_PY_PKG "${CMAKE_CURRENT_SOURCE_DIR}/arboocr")
add_custom_command(TARGET _arboocr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_ARBOOCR_PY_PKG}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_arboocr>
        "${_ARBOOCR_PY_PKG}/$<TARGET_FILE_NAME:_arboocr>"
    COMMENT "Copy _arboocr extension into python/arboocr/"
)

if(ARBOOCR_USE_SYSTEM_DEPS)
    set_target_properties(_arboocr PROPERTIES
        INSTALL_RPATH "${ARBOOCR_ORT_LIB_DIR}" BUILD_WITH_INSTALL_RPATH TRUE)
endif()
