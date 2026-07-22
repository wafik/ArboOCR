"""arboOCR Python bindings — Engine facade over the native C++ library."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_native_dlls() -> None:
    """On Windows, ensure OpenCV/ORT DLLs are findable when loading _arboocr."""
    if sys.platform != "win32":
        return
    # Explicit override for packagers / CI
    extra = os.environ.get("ARBOOCR_DLL_DIR", "").strip()
    candidates: list[Path] = []
    if extra:
        candidates.append(Path(extra))
    # Common vcpkg layout next to a CMake build tree (dev workflow)
    here = Path(__file__).resolve()
    repo = here.parents[2]  # .../python/arboocr -> repo root
    candidates.append(
        repo / "build" / "windows-x64" / "vcpkg_installed" / "x64-windows" / "bin"
    )
    candidates.append(
        repo / "build" / "linux-x64" / "vcpkg_installed" / "x64-linux" / "lib"
    )
    add = getattr(os, "add_dll_directory", None)
    for d in candidates:
        if d.is_dir():
            if add is not None:
                try:
                    add(str(d))
                except OSError:
                    pass
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            break


_bootstrap_native_dlls()

try:
    from arboocr._arboocr import (
        Engine,
        EngineConfig,
        LinePrediction,
        PagePrediction,
        Point2f,
        detect_cuda,
        detect_tensorrt,
        resolve_model_paths,
        to_json,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "arboOCR native extension not found or its DLLs failed to load. "
        "Build with -DARBOOCR_BUILD_PYTHON=ON, then set PYTHONPATH=python. "
        "On Windows, set ARBOOCR_DLL_DIR to the folder containing "
        "onnxruntime.dll and opencv_*.dll (e.g. build/.../vcpkg_installed/"
        "x64-windows/bin). Underlying error: " + str(e)
    ) from e

__all__ = [
    "Engine",
    "EngineConfig",
    "LinePrediction",
    "PagePrediction",
    "Point2f",
    "detect_cuda",
    "detect_tensorrt",
    "resolve_model_paths",
    "to_json",
]

__version__ = "0.1.0"
