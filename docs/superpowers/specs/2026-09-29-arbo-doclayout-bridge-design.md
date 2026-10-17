# arboOCR × arbo-doclayout bridge — design

**Date:** 2026-09-29
**Status:** approved (user: "lanjut semuanya dengan baik")
**Related:** `../arbo-doclayout` (sibling lib, verified working), `2026-07-23-accuracy-improve-design.md` ("revisit arbo-doclayout if reading-order failures appear"), braillify-core `feat/doclayout-reading-order` (same algorithm, Python reference).

## Problem

`sortLinesReadingOrder` (`src/arboOCR/ocr_utils.cpp:425`) sorts by centroid
y-row then x — no column awareness. Multi-column pages interleave line by
line (proven on Bobo magazine page-0010 in the Python prototype). Markdown
reconstruction (`markdown.cpp`) assumes reading-ordered input and inherits
the interleaving.

`sibling arbo-doclayout` already solves region detection:
`Engine::analyze(cv::Mat) → PageLayout`, boxes **pre-sorted by the model's
`reading_order` column**, warm latency **±441 ms** CPU, 13/13 tests green,
model auto-downloads (203 MB, `~/.cache/arbo-doclayout`). What is missing is
only the bridge: assign OCR lines to layout blocks and emit them in block
order.

## Goal

Correct reading order on multi-column / structured pages, **without changing
default behavior or cost for anyone who does not opt in.**

## Non-goals

- No plugin system in arboOCR (roadmap: single-purpose inference core;
  layout stays a separate repo).
- No changes to detector, classifier, recognizer, or `markdown.cpp`.
- No changes to `sortLinesReadingOrder` itself (it stays the fallback).
- No default-on in this cycle; no binding (Python/PHP) exposure yet.
- No fix for the ~611 benign ORT `Schema error` stderr lines emitted by the
  vcpkg onnxruntime build (pre-existing, cosmetic, out of scope).

## Two gates — both default OFF

| Gate | Layer | Default | Effect when off |
|---|---|---|---|
| `ARBOOCR_WITH_DOCLAYOUT` | CMake (compile/link) | `OFF` | Zero code, zero link, build identical to today |
| `EngineConfig::useDocLayout` | runtime (per-engine) | `false` | No `analyze()` call, no model download, no +441 ms, no +85 MB |

Release builds may later turn gate 1 ON while keeping gate 2 runtime —
end users then choose per call.

## Design

### CMake (arboOCR)

```cmake
option(ARBOOCR_WITH_DOCLAYOUT "Link sibling arbo-doclayout (block layout)" OFF)
set(ARBOOCR_DOCLAYOUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../arbo-doclayout"
    CACHE PATH "Path to arbo-doclayout sources")
if(ARBOOCR_WITH_DOCLAYOUT)
  set(ARBODOCLAYOUT_BUILD_TESTS OFF CACHE BOOL "" FORCE)  # our option, see below
  add_subdirectory(${ARBOOCR_DOCLAYOUT_DIR} ${CMAKE_BINARY_DIR}/arbo-doclayout)
endif()
```

Library links `arboDocLayout` **PUBLIC** only when the option is ON — the
public header `engine.hpp` includes `arboDocLayout/engine.hpp` under the
define, so tests/demo need the include path too. The define
(`ARBOOCR_HAS_DOCLAYOUT`) is PUBLIC as well, so tests see the gated config
field. All doclayout code in `src/` is guarded by that define so the OFF
build compiles none of it.

### arbo-doclayout tweak (small, separate commit)

- Add `option(ARBODOCLAYOUT_BUILD_TESTS "Build tests + demo" ON)` guarding
  the `tests/`, `cli/`, `doctest`/`cxxopts` `find_package` calls — so
  consumers are not forced to provide them. Default ON keeps standalone dev
  unchanged.

### EngineConfig

```cpp
#ifdef ARBOOCR_HAS_DOCLAYOUT
    // Run the sibling arbo-doclayout model before ordering lines ...
    // Costs ~+440 ms/image, +85 MB RSS, 203 MB one-time model download.
    // Fallback to sortLinesReadingOrder on any failure or zero blocks.
    bool useDocLayout = false;
#endif
```

Config field is visible only when the build has the capability — a build
without the library cannot be tricked into using it.

### Pipeline (`Engine::recognize`)

Current end: build `result.lines` → `sortLinesReadingOrder(result.lines)`.

New:

```
if (useDocLayout) {                       // inside try/catch, fail-soft
    PageLayout page = layoutEngine().analyze(srcImage);   // never throws
    if (!page.boxes.empty() && !result.lines.empty())
        orderLinesByBlocks(result.lines, page.boxes);     // replaces sort
    else
        sortLinesReadingOrder(result.lines);
} else {
    sortLinesReadingOrder(result.lines);
}
```

- `layoutEngine()`: lazily-constructed `arbo::doclayout::Engine` singleton
  per `Engine` instance, built on first `useDocLayout=true` call. Its
  **constructor can throw** (bad/missing model) → catch, log once, disable
  the flag for that instance (permanent fail-soft, mirrors braillify
  `_failed`).
- Image passed to `analyze` is the same `cv::Mat` already in memory (the
  `PagePrediction::image` source), no re-read.

### `orderLinesByBlocks` (new, in `ocr_utils` — pure function, no ORT)

Port of the algorithm validated in Python (`test-ocr-doclayout`) and
braillify (`_layout_reading_order`), simplified because PP-DocLayoutV2
already returns blocks in reading order:

1. **Bucket:** each line → block with max (intersection / line area);
   lines touching no block (orphans) → nearest block by center distance.
2. **Emit:** blocks in vector order (= model `reading_order`); inside a
   block, lines sorted y then x.
3. **Header/footer pin:** if any block label is `header`/`header_image`,
   it is emitted first anyway (model usually does this; do not second-guess
   — only reorder if a header-labelled block is not already before all
   content blocks). Same for `footer` last. Keep this as a cheap
   normalize pass, not a re-sort.

No column clustering needed — the model's order is the source of truth
(verified on page-0010: header → title → left column → right column →
image → bottom).

### Label mapping

arbo-doclayout uses PP-DocLayout labels (`text`, `paragraph_title`,
`doc_title`, `header`, `header_image`, `footer`, …). Header pin matches
`header`, `header_image` (prefix `header`); footer pin matches `footer`.
No DocLayout-YOLO labels involved (that was the Python prototype only).

### Errors / fail-soft matrix

| Condition | Behavior |
|---|---|
| Built without option | `useDocLayout` field absent; old path only |
| Model download/load fails (ctor throws) | log once, flag → false forever, old sort |
| `analyze` returns empty boxes | old sort |
| Zero lines | nothing to order |
| Any exception inside ordering | catch → old sort |

OCR output is never worse than today, by construction.

## Testing

1. **`tests/test_ocr_utils.cpp` — synthetic order tests (no model, no ORT):**
   `orderLinesByBlocks` with hand-built boxes:
   - two-column interleaved lines + two column blocks → left run then right run
   - orphan line (no overlap) → nearest block
   - header block emitted first, footer last
   - empty blocks vector → caller falls back (assert function returns false
     or similar sentinel so `recognize` picks fallback)
2. **`tests/test_engine.cpp` — flag tests:** `useDocLayout=false` (default)
   path unchanged; config field only exists with `ARBOOCR_HAS_DOCLAYOUT`.
3. **Inference smoke** (optional, gated like `test_engine_inference.cpp`):
   if model cache present, page-0010 → block count > 0. Skipped when absent —
   tests never download.

## Cost (measured / verified)

- Runtime when enabled: **±441 ms** + **±85 MB RSS** per process, **203 MB
  one-time** model download (auto, first analyze).
- When disabled (default): **0**.
- Build with option ON adds sibling sources + needs the model only at
  runtime.

## Files touched

| Repo | File | Change |
|---|---|---|
| arbo-doclayout | `CMakeLists.txt` | `ARBODOCLAYOUT_BUILD_TESTS` option |
| arboOCR | `CMakeLists.txt` | option + subdirectory + link + define |
| arboOCR | `include/arboOCR/engine.hpp` | gated `useDocLayout` field |
| arboOCR | `include/arboOCR/ocr_utils.hpp` | `orderLinesByBlocks` decl |
| arboOCR | `src/arboOCR/ocr_utils.cpp` | `orderLinesByBlocks` impl |
| arboOCR | `src/arboOCR/engine.cpp` | gated branch + lazy engine + fail-soft |
| arboOCR | `tests/test_ocr_utils.cpp` | synthetic order tests |
| arboOCR | `README.md` | option/flag row in config docs |

## Resolved decisions

| Topic | Decision |
|---|---|
| New plugin? | No — sibling lib + optional link (user: "terpisah namun bisa disambungkan") |
| Default | gate `OFF`, flag `false` (user-approved) |
| Ordering source | model `reading_order` (vector order), not heuristic column sort |
| Fallback | existing `sortLinesReadingOrder`, always |
| Scope this cycle | library + tests + README; no bindings |
