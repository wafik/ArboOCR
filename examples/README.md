# arboOCR examples

Two small, buildable programs — distinct from [`cli/arboocr_demo.cpp`](../cli/arboocr_demo.cpp)
(a full-featured CLI tool with argument parsing for every `EngineConfig`
field) and from the code snippets in the main [README](../README.md) (which
aren't compiled, just illustrative). These are built as part of the normal
project build — no separate setup needed.

## `basic_recognize` — the common case

Construct an `Engine`, call `recognize()`, print each line's text, score,
and bounding polygon, with a simple empty-result check. This is exactly
the pattern in the README's Quickstart section, made buildable.

```bash
./arboocr_example_basic path/to/image.jpg [models-dir] [--tensorrt]
```

Defaults to CPU. Pass `--tensorrt` to opt in — the *first* run against a
given model compiles a TensorRT engine from scratch, which can take
several minutes on modest hardware (e.g. a Jetson Nano); subsequent runs
reuse the cache (`EngineConfig::trtCacheDir`) and are fast. We hit this
ourselves while testing this example: leaving TensorRT on by default meant
the first run just... sat there for minutes with no output, which is a bad
first impression for something meant to demo instantly. CPU-first, opt
into the fast-but-slow-to-warm-up backend explicitly.

## `custom_pipeline` — driving the stages directly

`Engine::recognize()` runs every detected box through recognition
unconditionally. This example shows why and how you'd bypass that: it
drives `Detector` → filter-by-score → `Classifier` → `Recognizer`
manually, dropping low-confidence detections *before* paying for the (much
more expensive) recognition pass — something `Engine` has no hook for.

```bash
./arboocr_example_custom_pipeline path/to/image.jpg [models-dir] [min-box-score]
```

Read the comments in [`custom_pipeline.cpp`](custom_pipeline.cpp) — each
step is annotated with which `Engine::recognize()` step it corresponds to
(see [`src/arboOCR/engine.cpp`](../src/arboOCR/engine.cpp) for the original)
and what's different.

## Building

These are built automatically as part of the normal project build (see the
main [README](../README.md#build)):

```bash
cmake --build build/<preset>
```

Binaries land next to the other build outputs
(`arboocr_tests`, `arboocr_demo`) in your build directory's output folder.
Both examples need a populated `models/` directory — see the main README's
[Models](../README.md#models) section.
