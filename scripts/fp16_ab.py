#!/usr/bin/env python3
"""A/B TensorRT FP16 against FP32, using CPU output as the reference.

Roadmap item #14: EngineConfig::useFp16 defaults to true, and nothing
exercises it. RapidOCR measured TensorRT FP16 collapsing PP-OCRv4/v5
det_server models (H-mean 0.8161 -> 0.0905) with no diagnosis. arboOCR
ships a single non-server detector so it is probably in the safe class —
"probably" is why this script exists.

No ground-truth labels needed: the CPU path is the well-tested one, so we
ask "does TensorRT agree with CPU, and does FP16 agree less than FP32?".
That is the question the default actually rides on.

Usage:
    python scripts/fp16_ab.py --images path/to/dir --models-dir models

Each arm gets its own --trt-cache-dir. Mixing them silently loads an
engine built for the other precision, which would make the whole
comparison meaningless.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run_arm(binary, image, models_dir, model_type, arm, cache_root):
    """Run one image through one arm. Returns (lines, elapsed_ms, backend) or None.

    The backend is returned because Engine auto-falls back TensorRT -> CUDA
    -> CPU. On a box without TensorRT every arm quietly runs the same CPU
    code and agrees with itself 100%, which reads as "FP16 is fine" when
    nothing was tested at all. See check_backends().
    """
    cmd = [
        str(binary),
        "--image", str(image),
        "--models-dir", str(models_dir),
        "--model-type", model_type,
        "--json",
    ]
    if arm != "cpu":
        cmd += ["--tensorrt", "--trt-cache-dir", str(cache_root / arm)]
        cmd += ["--fp16=true" if arm == "trt-fp16" else "--fp16=false"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):  # 1 == no text found, still a valid result
        print(f"  ! {arm} failed on {image.name} (exit {proc.returncode})",
              file=sys.stderr)
        if proc.stderr.strip():
            print(f"    {proc.stderr.strip().splitlines()[-1]}", file=sys.stderr)
        return None
    try:
        page = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  ! {arm}: could not parse JSON for {image.name}", file=sys.stderr)
        return None
    return ([ln["text"] for ln in page.get("lines", [])],
            page.get("elapsedMs", 0.0),
            page.get("backend", "unknown"))


def check_backends(results):
    """Refuse to grade if a trt-* arm did not actually get TensorRT.

    This is the whole reason the script can be trusted: without it, running
    on a CPU-only machine prints a confident 100% PASS that means nothing.
    """
    bad = []
    for arm, per_image in results.items():
        if not arm.startswith("trt-") or not per_image:
            continue
        backends = {v[3] for v in per_image.values()}
        if backends != {"tensorrt"}:
            bad.append((arm, sorted(backends)))
    return bad


def similarity(a, b):
    """Char similarity between two pages, compared as one joined string.

    Joined rather than per-line because a precision change can alter the
    number of detected lines; comparing line-by-line would silently drop
    exactly the failure we are hunting for.
    """
    return SequenceMatcher(None, "\n".join(a), "\n".join(b)).ratio()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path,
                    help="Directory of images, or a single image file")
    ap.add_argument("--models-dir", default="models", type=Path)
    ap.add_argument("--model-type", default="small",
                    help="tiny|small|medium (default: small — the library default)")
    ap.add_argument("--bin", type=Path,
                    default=Path("build/windows-x64/Release/arboocr_demo.exe")
                    if sys.platform == "win32" else Path("build/jetson/arboocr_demo"))
    ap.add_argument("--cache-root", default=Path("build/trt_ab"), type=Path,
                    help="Parent for the per-arm TensorRT engine caches")
    ap.add_argument("--arms", default="cpu,trt-fp32,trt-fp16",
                    help="Comma-separated. Drop the trt-* arms to self-test on a CPU box.")
    ap.add_argument("--fail-under", type=float, default=0.98,
                    help="Exit 1 if mean FP16-vs-CPU similarity falls below this")
    args = ap.parse_args()

    if not args.bin.exists():
        sys.exit(f"binary not found: {args.bin} (pass --bin)")

    if args.images.is_dir():
        images = sorted(p for p in args.images.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        images = [args.images]
    if not images:
        sys.exit(f"no images found in {args.images}")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "cpu" not in arms:
        sys.exit("the cpu arm is the reference — it cannot be dropped")

    print(f"{len(images)} image(s), model-type={args.model_type}, arms={arms}")
    print("First TensorRT run per arm compiles and caches an engine — "
          "minutes on a Nano. Timings below are from the cached second pass.\n")

    results = {arm: {} for arm in arms}
    for image in images:
        for arm in arms:
            out = run_arm(args.bin, image, args.models_dir, args.model_type,
                          arm, args.cache_root)
            if out is None:
                continue
            # Second pass: engine is cached now, so this timing is the honest one.
            t0 = time.perf_counter()
            out2 = run_arm(args.bin, image, args.models_dir, args.model_type,
                           arm, args.cache_root)
            wall = (time.perf_counter() - t0) * 1000.0
            lines, elapsed, backend = out2 if out2 is not None else out
            results[arm][image.name] = (lines, elapsed, wall, backend)

    print(f"{'image':<28} " + " ".join(f"{a:>22}" for a in arms if a != "cpu"))
    print("-" * (28 + 23 * max(1, len(arms) - 1)))
    sims = {arm: [] for arm in arms if arm != "cpu"}
    for image in images:
        if image.name not in results["cpu"]:
            continue
        ref = results["cpu"][image.name][0]
        cells = []
        for arm in arms:
            if arm == "cpu":
                continue
            if image.name not in results[arm]:
                cells.append(f"{'--':>22}")
                continue
            got = results[arm][image.name][0]
            s = similarity(ref, got)
            sims[arm].append(s)
            cells.append(f"{s * 100:>15.2f}% {results[arm][image.name][1]:>5.0f}ms")
        print(f"{image.name:<28} " + " ".join(cells))

    print(f"\n{'arm':<12} {'mean sim vs cpu':>16} {'mean ms':>10} {'lines':>8}")
    for arm in arms:
        got = results[arm]
        if not got:
            print(f"{arm:<12} {'no results':>16}")
            continue
        ms = statistics.mean(v[1] for v in got.values())
        nlines = sum(len(v[0]) for v in got.values())
        if arm == "cpu":
            print(f"{arm:<12} {'(reference)':>16} {ms:>10.1f} {nlines:>8}")
        else:
            print(f"{arm:<12} {statistics.mean(sims[arm]) * 100:>15.2f}% "
                  f"{ms:>10.1f} {nlines:>8}")

    fell_back = check_backends(results)
    if fell_back:
        print()
        for arm, backends in fell_back:
            print(f"INCONCLUSIVE: {arm} ran on {'/'.join(backends)}, not tensorrt.")
        print("Engine auto-falls back TensorRT -> CUDA -> CPU, so these arms "
              "executed identical CPU code and agreeing 100% proves nothing "
              "about FP16. Re-run on a machine with a TensorRT-capable "
              "onnxruntime build (e.g. the Jetson).")
        return 2

    if "trt-fp16" in sims and sims["trt-fp16"]:
        mean16 = statistics.mean(sims["trt-fp16"])
        print()
        if mean16 < args.fail_under:
            print(f"FAIL: FP16 agrees with CPU only {mean16 * 100:.2f}% "
                  f"(threshold {args.fail_under * 100:.0f}%). "
                  f"useFp16=true is NOT a safe default for these models.")
            return 1
        print(f"PASS: FP16 agrees with CPU {mean16 * 100:.2f}%. "
              f"useFp16=true holds up on this set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
