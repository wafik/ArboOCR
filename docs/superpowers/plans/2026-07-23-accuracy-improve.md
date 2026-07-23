# arboOCR Accuracy & Latency Honesty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the measured accuracy gap vs ppu-paddle-ocr on SROIE receipts by measure-then-fix: honest rec scores, CPU-safe defaults, warm fair bench, det/rec diagnosis, one evidence-based pipeline fix, re-bench.

**Architecture:** Library changes in `arboOCR` (types/engine/CLI/Python/docs/tests) land first. Diagnosis and warm compare live in `cpp/compare` (Bun/TS + optional Python `arboocr.Engine`). Phase 4 pipeline fix is chosen only after Phase 3 writes a det/rec/mixed verdict — do not invent threshold changes before that.

**Tech Stack:** C++17, ONNXRuntime, OpenCV, doctest, pybind11, Bun/TypeScript compare harness, SROIE2019 under `cpp/compare/SROIE2019`.

**Spec:** `docs/superpowers/specs/2026-07-23-accuracy-improve-design.md`  
**Baseline:** `cpp/compare/RESULTS.md` (arbo small avg sim **84.1%**, ppu small **94.6%**)

## Global Constraints

- Do **not** integrate `arbo-doclayout` in this cycle.
- Do **not** default to medium for accuracy; default rec size is **small**.
- Do **not** force `det_medium` in harness unless that pair is the explicit experiment.
- CLAHE remains opt-in / separate design — enable only if Phase 3 proves low-contrast misses.
- Prefer fewest files (ponytail); collapse scripts if clearer.
- Existing `arboocr_tests` must stay green after each library task.
- Commit only if the tree is a git repo and the user wants commits; otherwise skip commit steps.

---

## File map

| Path | Role |
|------|------|
| `include/arboOCR/types.hpp` | `LinePrediction::detScore`; document `score` = rec confidence |
| `src/arboOCR/types.cpp` | JSON emit `detScore`; optional helper if placed here |
| `include/arboOCR/engine.hpp` | default `modelType = "small"`; comment det policy |
| `src/arboOCR/engine.cpp` | wire `score` from `charScores`, `detScore` from box |
| `cli/arboocr_demo.cpp` | print rec + det scores |
| `python/bindings/module.cpp` | `det_score` field |
| `python/tests/test_smoke.py` | defaults + `det_score` |
| `tests/test_engine.cpp` | defaults, JSON, score semantics |
| `README.md` | defaults + det-size policy |
| `cpp/compare/bench_warm.ts` | warm ppu + arbo (Python) compare |
| `cpp/compare/diag_det_rec.ts` | det/rec diagnosis → verdict |
| `cpp/compare/sweep_det.ts` | optional, only if det-bound |
| `cpp/compare/RESULTS.md` | before/after |
| Phase 4 files | TBD by verdict (likely `engine.hpp` thresholds and/or `recognizer.cpp`) |

---

### Task 1: Rec score + `detScore` on `LinePrediction`

**Files:**
- Modify: `include/arboOCR/types.hpp`
- Modify: `src/arboOCR/types.cpp`
- Modify: `src/arboOCR/engine.cpp`
- Modify: `tests/test_engine.cpp`
- Modify: `cli/arboocr_demo.cpp` (print both — small, same task)

**Interfaces:**
- Produces: `LinePrediction { polygon, text, score /*rec*/, detScore /*det*/ }`
- Produces: rec score = mean of `RawTextLine::charScores` (0 if empty)

- [ ] **Step 1: Write failing tests in `tests/test_engine.cpp`**

Update the defaults test later in Task 2. For this task, add/adjust:

```cpp
TEST_CASE("LinePrediction defaults include detScore") {
    LinePrediction lp;
    CHECK(lp.score == doctest::Approx(0.0f));
    CHECK(lp.detScore == doctest::Approx(0.0f));
}

TEST_CASE("toJson includes detScore") {
    LinePrediction line;
    line.text = "ab";
    line.score = 0.9f;
    line.detScore = 0.7f;
    line.polygon = {{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}, {7.f, 8.f}};
    const std::string json = toJson(line);
    CHECK(json.find("\"score\":0.9") != std::string::npos);
    CHECK(json.find("\"detScore\":0.7") != std::string::npos);
}

// Pure helper test — if meanRecScore is free function in types.hpp:
TEST_CASE("meanRecScore averages char scores and empty is 0") {
    CHECK(meanRecScore({}) == doctest::Approx(0.0f));
    CHECK(meanRecScore(std::vector<float>{0.5f, 1.0f}) == doctest::Approx(0.75f));
}
```

Fix existing aggregate init in `toJson serializes page...` test:

```cpp
page.lines.push_back(LinePrediction{
    {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}},
    "hello\nworld",
    0.91f,  // score (rec)
    0.0f,   // detScore
});
```

- [ ] **Step 2: Run tests — expect fail (no `detScore` / `meanRecScore`)**

```powershell
cmake --build D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64 --config Release --target arboocr_tests
# compile error or failed CHECKs
```

- [ ] **Step 3: Implement types**

In `include/arboOCR/types.hpp`, replace the line prediction block with:

```cpp
// One recognized text line: polygon, decoded text, recognition confidence,
// and detector box score.
struct LinePrediction {
    Polygon polygon;
    std::string text;
    /// Mean CTC character confidence from the recognizer (0 if no chars).
    float score = 0.0f;
    /// Detector box score for this polygon (0 if unknown).
    float detScore = 0.0f;
};

/// Mean of per-character CTC scores; empty → 0. Used to fill LinePrediction::score.
inline float meanRecScore(const std::vector<float>& charScores) {
    if (charScores.empty()) return 0.0f;
    float sum = 0.0f;
    for (float s : charScores) sum += s;
    return sum / static_cast<float>(charScores.size());
}
```

In `src/arboOCR/types.cpp` `appendLine`, emit both fields:

```cpp
// pretty branch: after "score":
<< padIn << "\"score\":" << line.score << ",\n"
<< padIn << "\"detScore\":" << line.detScore << ",\n"
// compact:
os << "{\"text\":\"" << escapeJson(line.text) << "\",\"score\":" << line.score
   << ",\"detScore\":" << line.detScore << ",\"polygon\":";
```

- [ ] **Step 4: Wire engine.cpp**

In `Engine::runPipeline`, replace the push_back loop:

```cpp
for (size_t i = 0; i < textBoxes.size(); i++) {
    const float recScore = (i < textLines.size())
        ? meanRecScore(textLines[i].charScores)
        : 0.0f;
    const std::string& text = (i < textLines.size()) ? textLines[i].text : std::string{};
    result.lines.push_back(LinePrediction{
        cvPointsToPolygon(textBoxes[i].boxPoint),
        text,
        recScore,
        textBoxes[i].score,
    });
}
```

Include nothing extra if `meanRecScore` is in `types.hpp` (engine already includes types via engine.hpp chain — ensure `types.hpp` is visible).

- [ ] **Step 5: CLI print both scores**

In `cli/arboocr_demo.cpp` line print loop, change score print to:

```cpp
std::cout << "  [" << i << "] \"" << line.text
          << "\" (score=" << line.score
          << " detScore=" << line.detScore << ")";
```

- [ ] **Step 6: Build & run tests**

```powershell
cmake --build D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64 --config Release --target arboocr_tests
cd D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\Release
$env:PATH = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\Release;D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\vcpkg_installed\x64-windows\bin;" + $env:PATH
.\arboocr_tests.exe
```

Expected: all passed.

- [ ] **Step 7: Commit (if git)**

```text
fix(arboOCR): LinePrediction.score is rec conf; add detScore
```

---

### Task 2: Default `modelType` = `small` + docs

**Files:**
- Modify: `include/arboOCR/engine.hpp` (`modelType` default + comment)
- Modify: `tests/test_engine.cpp` (default expectation)
- Modify: `python/tests/test_smoke.py` (`model_type == "small"`)
- Modify: `README.md` (Choosing a model size / defaults)

**Interfaces:**
- Produces: `EngineConfig{}.modelType == "small"`
- Det resolution unchanged: still `<ocrVersion>_det.onnx`

- [ ] **Step 1: Fail the defaults test**

Change in `tests/test_engine.cpp`:

```cpp
CHECK(cfg.modelType == "small"); // was medium; CPU-honest default
```

In `python/tests/test_smoke.py`:

```python
self.assertEqual(cfg.model_type, "small")
```

- [ ] **Step 2: Run C++ test — expect fail on modelType**

```powershell
# rebuild + arboocr_tests → FAIL EngineConfig has sane defaults
```

- [ ] **Step 3: Change default in `engine.hpp`**

```cpp
// Default "small": best CPU accuracy/latency tradeoff measured on SROIE
// smoke compare (medium ~4× slower on CPU with no sim gain on that set).
// Det is always "<ocrVersion>_det.onnx" unless detModelPath is set — modelType
// does NOT select a det size variant.
std::string modelType = "small";
```

Update any doc comment above that still says default medium.

- [ ] **Step 4: README — short policy**

In Models / Choosing a model size section, add:

```markdown
**Defaults:** `modelType` defaults to `small` (CPU). Prefer `tiny` for
throughput, `medium` only when you have GPU/TensorRT headroom and have
measured a real accuracy win on *your* data. Detection always loads
`<ocrVersion>_det.onnx` unless you set `detModelPath` (there is no
`modelType`-selected det file).
```

- [ ] **Step 5: Rebuild tests + Python smoke if bindings rebuilt**

```powershell
cmake --build ... --target arboocr_tests
.\arboocr_tests.exe
# if Python module rebuilt:
$env:PYTHONPATH = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\python"
$env:ARBOOCR_DLL_DIR = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\vcpkg_installed\x64-windows\bin"
python -m unittest python.tests.test_smoke -v
```

Note: Python tests only update after `_arboocr` is rebuilt (`ARBOOCR_BUILD_PYTHON=ON`). If not rebuilding Python this step, still change `test_smoke.py` so next bind build is correct; run unittest when possible.

- [ ] **Step 6: Commit**

```text
fix(arboOCR): default modelType to small for CPU honesty
```

---

### Task 3: Python bindings expose `det_score`

**Files:**
- Modify: `python/bindings/module.cpp`
- Modify: `python/tests/test_smoke.py`

- [ ] **Step 1: Extend smoke test**

```python
def test_line_prediction_fields(self):
    line = LinePrediction()
    line.text = "hi"
    line.score = 0.9
    line.det_score = 0.8
    line.polygon = [Point2f(1.0, 2.0), Point2f(3.0, 4.0)]
    self.assertEqual(line.text, "hi")
    self.assertAlmostEqual(line.score, 0.9)
    self.assertAlmostEqual(line.det_score, 0.8)
    self.assertEqual(len(line.polygon), 2)
```

- [ ] **Step 2: Bind field**

```cpp
py::class_<LinePrediction>(m, "LinePrediction")
    .def(py::init<>())
    .def_readwrite("polygon", &LinePrediction::polygon)
    .def_readwrite("text", &LinePrediction::text)
    .def_readwrite("score", &LinePrediction::score)
    .def_readwrite("det_score", &LinePrediction::detScore);
```

- [ ] **Step 3: Rebuild Python extension + run smoke**

```powershell
cmake --build D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64 --config Release --target _arboocr
# copy pyd if build tree differs from python/arboocr/ (project already places/copies — follow existing layout)
$env:PYTHONPATH = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\python"
$env:ARBOOCR_DLL_DIR = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\vcpkg_installed\x64-windows\bin"
python -m unittest discover -s D:\kerjaan\kreasi\kimfu\cpp\arboOCR\python\tests -v
```

Expected: PASS (or skip if import fails — then fix DLL path first).

- [ ] **Step 4: Commit**

```text
feat(arboOCR): expose LinePrediction.det_score in Python bindings
```

---

### Task 4: Warm fair compare harness (`bench_warm.ts`)

**Files:**
- Create: `D:/kerjaan/kreasi/kimfu/cpp/compare/bench_warm.ts`
- Uses: existing ppu import path, SROIE images, arbo Python Engine

**Interfaces:**
- Consumes: `PaddleOcrService`, `V6_TINY_MODEL`, `V6_SMALL_MODEL` from ppu
- Consumes: `arboocr.Engine` with long-lived instance per size
- Produces: `cpp/compare/out/bench_warm.json` + console table (ms + sim)

- [ ] **Step 1: Implement `bench_warm.ts`**

```typescript
// ponytail: warm ppu vs arbo (Python Engine) on SROIE sample
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { join, basename } from "path";
import { spawnSync } from "child_process";
import {
  PaddleOcrService,
  V6_TINY_MODEL,
  V6_SMALL_MODEL,
  type ModelUrls,
} from "../ppu-paddle-ocr/src/index.ts";

const ROOT = import.meta.dir;
const IMG_DIR = join(ROOT, "SROIE2019/test/img");
const BOX_DIR = join(ROOT, "SROIE2019/test/box");
const OUT = join(ROOT, "out");
const SIZES = ["tiny", "small"] as const;
const SAMPLES = [
  "X00016469670.jpg",
  "X00016469671.jpg",
  "X51005200931.jpg",
  "X51005230605.jpg",
  "X51005230616.jpg",
];
const ARBO_MODELS =
  process.env.ARBO_MODELS ?? "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/models";
const ARBO_PYTHON =
  process.env.ARBO_PYTHON ?? "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/python";
const ARBO_DLL =
  process.env.ARBOOCR_DLL_DIR ??
  "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/build/windows-x64/vcpkg_installed/x64-windows/bin";

const PPU: Record<(typeof SIZES)[number], ModelUrls> = {
  tiny: V6_TINY_MODEL,
  small: V6_SMALL_MODEL,
};

function gtText(stem: string): string {
  const p = join(BOX_DIR, `${stem}.txt`);
  if (!existsSync(p)) return "";
  return readFileSync(p, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((l) => l.split(",").slice(8).join(",").trim())
    .join("\n");
}

function similarity(a: string, b: string): number {
  a = a.replace(/\s+/g, " ").trim().toLowerCase();
  b = b.replace(/\s+/g, " ").trim().toLowerCase();
  if (!a && !b) return 1;
  if (!a || !b) return 0;
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
  return 1 - dp[m][n] / Math.max(m, n);
}

/** One Python process: load Engine once, OCR all images, print JSON lines. */
function runArboWarm(size: string, images: string[]) {
  const py = `
import json, sys, os
os.environ["ARBOOCR_DLL_DIR"] = ${JSON.stringify(ARBO_DLL)}
sys.path.insert(0, ${JSON.stringify(ARBO_PYTHON)})
from arboocr import Engine, EngineConfig
cfg = EngineConfig()
cfg.models_dir = ${JSON.stringify(ARBO_MODELS)}
cfg.model_type = ${JSON.stringify(size)}
cfg.use_cuda = False
cfg.use_tensorrt = False
# default det file (no det_size force)
eng = Engine(cfg)
print(json.dumps({"event":"ready","backend":eng.backend()}), flush=True)
for path in ${JSON.stringify(images)}:
    page = eng.recognize(path)
    print(json.dumps({
        "image": path,
        "ms": page.elapsed_ms,
        "lines": len(page.lines),
        "text": "\\n".join(l.text for l in page.lines),
    }), flush=True)
`;
  const r = spawnSync("python", ["-c", py], {
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, ARBOOCR_DLL_DIR: ARBO_DLL, PYTHONPATH: ARBO_PYTHON },
  });
  if (r.status !== 0) {
    throw new Error(`arbo python failed: ${r.stderr || r.stdout}`);
  }
  const rows: { image: string; ms: number; lines: number; text: string }[] = [];
  for (const line of (r.stdout || "").split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const o = JSON.parse(line);
      if (o.event === "ready") console.log("arbo backend", o.backend, "size", size);
      if (o.image) rows.push(o);
    } catch { /* ignore non-json noise */ }
  }
  return rows;
}

mkdirSync(OUT, { recursive: true });
const results: unknown[] = [];

for (const size of SIZES) {
  console.log("\n=== ppu", size, "(warm service) ===");
  const svc = new PaddleOcrService({ model: PPU[size] });
  await svc.initialize();
  for (const name of SAMPLES) {
    const path = join(IMG_DIR, name);
    const gt = gtText(basename(name, ".jpg"));
    const buf = await Bun.file(path).arrayBuffer();
    const t0 = performance.now();
    const r = await svc.recognize(buf, { noCache: true });
    const ms = performance.now() - t0;
    const sim = similarity(r.text, gt);
    console.log(name, "ms", ms.toFixed(0), "sim", (sim * 100).toFixed(1) + "%");
    results.push({ engine: "ppu", size, image: name, ms, sim, text: r.text.trim() });
  }
  await svc.destroy();

  console.log("\n=== arbo", size, "(warm Engine) ===");
  const images = SAMPLES.map((n) => join(IMG_DIR, n));
  const arboRows = runArboWarm(size, images);
  for (const row of arboRows) {
    const name = basename(row.image);
    const gt = gtText(basename(name, ".jpg"));
    const sim = similarity(row.text, gt);
    console.log(name, "ms", row.ms, "sim", (sim * 100).toFixed(1) + "%");
    results.push({ engine: "arbo", size, image: name, ms: row.ms, sim, lines: row.lines, text: row.text });
  }
}

writeFileSync(join(OUT, "bench_warm.json"), JSON.stringify(results, null, 2));
console.log("wrote", join(OUT, "bench_warm.json"));
```

- [ ] **Step 2: Run**

```powershell
$env:PATH = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\vcpkg_installed\x64-windows\bin;" + $env:PATH
$env:PYTHONPATH = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\python"
$env:ARBOOCR_DLL_DIR = "D:\kerjaan\kreasi\kimfu\cpp\arboOCR\build\windows-x64\vcpkg_installed\x64-windows\bin"
cd D:\kerjaan\kreasi\kimfu\cpp\compare
bun bench_warm.ts
```

Expected: JSON rows for ppu/arbo × tiny/small × 5 images. If Python Engine fails, fix DLL path; last resort document and use multi-call demo (note spawn bias).

- [ ] **Step 3: Commit compare script**

```text
chore(compare): add warm ppu vs arbo bench harness
```

---

### Task 5: Diagnosis harness (`diag_det_rec.ts`)

**Files:**
- Create: `D:/kerjaan/kreasi/kimfu/cpp/compare/diag_det_rec.ts`
- Produces: `out/diag_verdict.md` + `out/diag.json`

**What to measure (minimum viable diagnosis):**

For each of the 5 images, using **arbo Python Engine** (or demo parse if needed):

1. Full-page sim vs GT (already have).
2. Line count arbo vs GT line count (`box` file line count).
3. Optional IoU: parse SROIE box quads; match to arbo polygons (IoU ≥ 0.5); count matched / GT / pred.
4. **Error samples:** list 10 worst GT lines (by edit distance to best-matched pred).

**Verdict rules (write into `diag_verdict.md`):**

```text
if mean |pred_lines - gt_lines| / gt_lines > 0.25 OR mean IoU-match recall < 0.7:
  → det-bound or mixed (det weak)
elif line counts close AND many matched boxes have high char errors:
  → rec-bound
else:
  → mixed
```

Also compare: does sim rise when using only matched lines? If yes, det noise/order is part of the gap.

- [ ] **Step 1: Implement diag script**

Keep it one file. Reuse `similarity` / `gtText` from Task 4 (copy — no shared package needed).

IoU for axis-aligned approx from polygon min/max is enough for SROIE (ponytail: AABB IoU first):

```typescript
function aabb(poly: { x: number; y: number }[]) {
  const xs = poly.map((p) => p.x), ys = poly.map((p) => p.y);
  return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
}
function iou(a: ReturnType<typeof aabb>, b: ReturnType<typeof aabb>) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const u = areaA + areaB - inter;
  return u <= 0 ? 0 : inter / u;
}
```

Python side should print per-line JSON: `text`, `score`, `det_score`, `polygon:[{x,y}...]` via `to_json` or manual.

Example Python snippet for one image:

```python
page = eng.recognize(path)
print(json.dumps({
  "image": path,
  "ms": page.elapsed_ms,
  "lines": [{
    "text": l.text,
    "score": l.score,
    "det_score": getattr(l, "det_score", 0.0),
    "polygon": [{"x": p.x, "y": p.y} for p in l.polygon],
  } for l in page.lines]
}))
```

- [ ] **Step 2: Run and write verdict**

```powershell
cd D:\kerjaan\kreasi\kimfu\cpp\compare
bun diag_det_rec.ts
```

Expected: `out/diag_verdict.md` ends with one of: `VERDICT: det-bound` | `rec-bound` | `mixed` plus numbers.

- [ ] **Step 3: Commit**

```text
chore(compare): add det/rec diagnosis harness + verdict
```

---

### Task 6: Evidence-based pipeline fix (Phase 4)

**Do not start until Task 5 verdict exists.**

**Files:** depend on verdict:

| Verdict | Touch |
|---------|--------|
| det-bound | `include/arboOCR/engine.hpp` default thresholds and/or README recommended values; optional small grid via `sweep_det.ts` then lock defaults |
| rec-bound | `src/arboOCR/recognizer.cpp` / dict handling — only with a failing case from diag |
| mixed | Prefer **one** det default change **or** one rec fix that moves the metric; avoid shotgun |

**Success bar:** re-run warm or `compare_ocr.ts` on 5 images → arbo **small** mean sim **> 84.1%**.

- [ ] **Step 1: Read `out/diag_verdict.md` and pick ONE fix class**

Write a 5-line note at top of the PR/commit body: verdict + change + why.

- [ ] **Step 2: If det-bound — optional micro-sweep then set defaults**

Create `cpp/compare/sweep_det.ts` only if needed: loop a **small** grid, e.g.

```text
detThresh ∈ {0.25, 0.3, 0.35}
detBoxThresh ∈ {0.5, 0.6}
detUnclipRatio ∈ {1.5, 1.6, 1.8}
```

Use Python Engine with overridden config fields each run (long-lived per param set). Pick best mean sim on 3 train images; verify on 2 holdout (`X51005230616`, `X51005200931` or similar).

If best defaults differ from `0.3 / 0.5 / 1.6`, update `engine.hpp` defaults **only if holdout does not regress**.

- [ ] **Step 3: If rec-bound — fix the concrete bug**

Examples (only if evidence points here):

- Empty `charScores` / dict load failure path
- Space/CTC edge case visible in error samples
- Preprocess mismatch on a reproduced crop

Add a unit test that fails before the fix when possible.

- [ ] **Step 4: Rebuild + re-bench**

```powershell
cmake --build ... --target arboocr_tests arboocr_demo _arboocr
.\arboocr_tests.exe
cd D:\kerjaan\kreasi\kimfu\cpp\compare
bun bench_warm.ts
# or bun compare_ocr.ts for full size matrix
```

Confirm mean sim(arbo small) > 0.841.

- [ ] **Step 5: Commit**

```text
fix(arboOCR): <specific fix from diagnosis>
```

---

### Task 7: Update RESULTS.md (before/after)

**Files:**
- Modify: `D:/kerjaan/kreasi/kimfu/cpp/compare/RESULTS.md`

- [ ] **Step 1: Append section**

```markdown
## After accuracy-improve cycle (YYYY-MM-DD)

### Changes landed
- score = rec mean CTC; detScore added
- default modelType = small
- diagnosis verdict: <det|rec|mixed>
- pipeline fix: <one line>

### Warm bench (tiny/small)
| engine | size | avg ms | avg sim |
...

### vs baseline
| | baseline arbo small | after |
|--|--:|--:|
| sim | 84.1% | X% |
| ms (warm or engine) | ... | ... |

### Remaining gap vs ppu
...
### Follow-ups
- larger sample, entity F1, GPU, doclayout only if reading-order issues
```

- [ ] **Step 2: Commit**

```text
docs(compare): RESULTS after arboOCR accuracy-improve cycle
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| score = rec, detScore | Task 1 |
| default modelType small + det policy docs | Task 2 |
| Python det_score | Task 3 |
| Warm fair compare | Task 4 |
| Diagnosis det/rec/mixed | Task 5 |
| One proven fix | Task 6 |
| Re-bench RESULTS | Task 7 |
| No doclayout / no medium-as-fix / CLAHE optional | Global constraints + Task 6 gates |
| Stretch ≥89% | Task 6 success is >84.1%; stretch noted in RESULTS not a hard fail |

**Placeholder scan:** Phase 4 file list is intentionally verdict-dependent; steps still specify how to choose and verify. No TBD left for Phases 1–3.

**Type consistency:** `detScore` (C++) / `det_score` (Python) / JSON `detScore`; `meanRecScore(vector<float>)→float`; `modelType` default `"small"`.

---

## Execution handoff

Plan complete and saved to:

`D:\kerjaan\kreasi\kimfu\cpp\arboOCR\docs\superpowers\plans\2026-07-23-accuracy-improve.md`

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session, `executing-plans`, batch with checkpoints  

Which approach?
