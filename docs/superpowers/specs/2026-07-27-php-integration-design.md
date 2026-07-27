# arboOCR → PHP integration design

Date: 2026-07-27

## Goal

Let a PHP application run arboOCR without building or vendoring the C++ source
tree. A framework-agnostic Composer package (`arbo-ocr-php`) shells out to a
**prebuilt, standalone** `arboocr_demo` binary produced by arboOCR's own CI and
published to GitHub Releases. PHP never sees CMake, vcpkg, or the repo — only a
downloaded binary folder.

Two independent components, built in order:

- **A. arboOCR (this repo):** add a machine-readable `--json` output to the CLI,
  and a GitHub Actions workflow that builds standalone Windows + Linux binaries
  and publishes them as release assets.
- **B. arbo-ocr-php (new repo at `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php`):**
  a Composer package that auto-downloads the matching release binary and runs it
  via `proc_open`, parsing the `--json` output.

Integration approach chosen: **CLI exec** (not FFI). No new C ABI to maintain;
per-image process spawn is negligible next to OCR inference time (seconds-scale).
The PHP package depends only on the *build artifact*, never on the arboOCR repo.

## Non-goals

- No FFI / shared-library ABI.
- No ONNX models in the release artifact (models are large and change
  independently; the consumer supplies `--models-dir`). arboOCR already has
  `downloadOcrModels()` for fetching them.
- No Jetson/aarch64 CI (needs a real aarch64 runner; not a GitHub-hosted target).
- No Laravel/WordPress-specific packaging in v1 (framework-agnostic library only;
  those can wrap it later).

---

## Component A — arboOCR changes

### A1. `--json` flag on `arboocr_demo`

`cli/arboocr_demo.cpp` gains a `--json` bool flag (default false). Behaviour:

- **JSON mode on:** stdout carries **only** JSON — a single object, nothing else.
  The human-readable `Backend:`/`Image:`/`Lines:` lines are suppressed so PHP can
  `json_decode()` the entire stdout stream.
- **JSON mode off:** unchanged current behaviour.

Output shape (one line, or pretty with an optional future flag — PHP only needs
compact):

```json
{
  "backend": "cpu",
  "image": "page.jpg",
  "elapsedMs": 402.05,
  "lines": [
    {"text": "INVOICE", "score": 0.845, "detScore": 0.91,
     "polygon": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
  ]
}
```

Reuse the existing `arbo::ocr::toJson(const PagePrediction&)` (in
`src/arboOCR/types.cpp`) for the `image`/`elapsedMs`/`lines` portion. `backend`
is **not** part of `PagePrediction` — it is only available via
`engine.backend()`. Fold it in one of two ways (implementer picks the smaller
diff):

- splice `"backend":"<x>",` into the string returned by `toJson`, or
- add an overload `toJson(page, backend, pretty)` in `types.cpp`.

Empty `lines` (no text found) is still valid JSON and still exit code 0 — matches
the library contract that `recognize()` never throws and an empty page is normal.

**Error contract:** if the engine can't be constructed (bad models dir, missing
ONNX) or any exception escapes, in JSON mode print a one-line message to
**stderr** and exit non-zero, leaving stdout empty. This gives PHP an
unambiguous success/failure signal (exit code + empty stdout) without parsing
prose.

**Check:** the existing `arboocr_tests` covers `toJson`. Add nothing heavy — a
single assertion (or a manual smoke run documented in the plan) that `--json`
stdout parses as JSON and contains `backend`/`lines` keys is enough.

### A2. Release workflow (`.github/workflows/release.yml`)

No CI exists in the repo today. Add one workflow.

- **Trigger:** `on: push: tags: ['v*']`.
- **Matrix jobs:**
  - `windows-x64` on `windows-latest`, preset `windows-x64` (MSVC + vcpkg).
  - `linux-x64` on `ubuntu-latest`, preset `linux-x64` (Ninja + vcpkg).
- **Per job:**
  1. Checkout.
  2. Bootstrap vcpkg (clone + `bootstrap-vcpkg`, or the `run-vcpkg` action),
     set `VCPKG_ROOT`. vcpkg installs onnxruntime/opencv/curl/doctest/cxxopts
     from `vcpkg.json`. (First run is slow — building OpenCV/onnxruntime from
     source. Enable vcpkg binary caching keyed on `vcpkg.json` to speed reruns;
     acceptable to leave cold in v1 if caching adds too much complexity.)
  3. `cmake --preset <name>`.
  4. `cmake --build build/<name> --config Release --target arboocr_demo`
     (only the CLI target — no need to build tests/examples/python for a release).
- **Package (flat, self-contained folder):**
  - Copy `arboocr_demo(.exe)` plus every non-system shared library it loads:
    - **Windows:** everything vcpkg already places next to the exe in
      `build/windows-x64/Release/` (e.g. `onnxruntime.dll`, `opencv_*.dll`,
      `libcurl.dll`, `abseil_dll.dll`, `re2.dll`, image-codec DLLs). Copy the
      whole set of DLLs sitting beside the exe.
    - **Linux:** run `ldd` on the binary, copy the `.so` files resolving outside
      `/lib`/`/usr/lib` (the vcpkg-built onnxruntime/opencv/curl and their
      deps) into the folder, and set `RPATH`/`$ORIGIN` (or ship a launcher) so
      the binary finds them next to itself.
  - Archive: `arboocr-windows-x64.zip`, `arboocr-linux-x64.tar.gz`.
- **Publish:** create/attach to the GitHub Release for the pushed tag and upload
  both archives (use `softprops/action-gh-release` or `gh release`). Asset names
  are version-free; the tag in the download URL is the version.

Resulting download URLs (what the PHP installer targets):

```
https://github.com/wafik/ArboOCR/releases/download/<tag>/arboocr-windows-x64.zip
https://github.com/wafik/ArboOCR/releases/download/<tag>/arboocr-linux-x64.tar.gz
```

---

## Component B — `arbo-ocr-php` package

New repo/dir: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php`. Composer package,
PSR-4, namespace `Arbo\Ocr`. Requires PHP with `ext-json`; installer needs
outbound HTTPS + a zip/tar extractor (`ext-zip`, or the `unzip`/`tar` CLI).

### B1. `composer.json`

- `name`: `arbo/ocr-php` (or user's preferred vendor).
- `autoload` PSR-4 `Arbo\\Ocr\\` → `src/`.
- `scripts`: `post-install-cmd` and `post-update-cmd` → `Arbo\Ocr\Installer::run`.
- `extra.arboocr-version`: the pinned arboOCR release tag the installer pulls
  (e.g. `v0.1.0`). Single source of truth for which binary matches this package
  version.
- `require-dev`: phpunit.

### B2. `Installer`

Static entry invoked by Composer after install/update.

1. Read the pinned tag from the package's own `composer.json` `extra`.
2. Detect platform via `PHP_OS_FAMILY` → `windows-x64` (Windows) or `linux-x64`
   (Linux). Unsupported OS (Darwin/BSD): print a clear notice and skip (no macOS
   artifact in v1) rather than failing the whole `composer install`.
3. Target dir: `bin/<platform>/` inside the package. If the expected binary is
   already there, skip (idempotent).
4. Download the matching asset from
   `https://github.com/wafik/ArboOCR/releases/download/<tag>/<asset>`.
5. Extract into `bin/<platform>/`; on Linux `chmod +x` the binary.
6. On any failure (network, 404, extract error): print an actionable message
   (including the manual-download URL and how to set `binPath` by hand) and do
   **not** hard-fail composer — the package is still usable with a
   user-supplied binary.

### B3. `Engine`

Public API the consumer uses.

- Constructor takes an options array / config object:
  - `binPath` (optional) — override the vendored binary path.
  - `modelsDir` (required for real use) — passed as `--models-dir`.
  - `ocrVersion`, `modelType`, `useAngleCls`, `useCuda`, `useTensorrt`,
    `useFp16`, `useClahe`, and the per-model path overrides — mapped 1:1 to the
    existing CLI flags.
  - Defaults `binPath` to `bin/<platform>/arboocr_demo(.exe)`.
- `recognize(string $imagePath): PageResult`:
  - Builds an **argv array** (not a shell string) — image paths and dirs are
    passed as separate argv elements, so no shell-injection surface.
  - Always appends `--json`.
  - Runs via `proc_open` with piped stdout/stderr; captures both, waits, reads
    exit code.
  - Exit 0 → `json_decode` stdout → `PageResult`. Empty `lines` is a normal
    result, not an error.
  - Non-zero exit, empty/ò unparseable stdout, or missing binary → throw
    `OcrException` carrying exit code + stderr text.

### B4. Value objects

- `PageResult`: `backend`, `image`, `elapsedMs`, `LineResult[] $lines`.
- `LineResult`: `text`, `score`, `detScore`, `Point[] $polygon` (or plain
  arrays — keep it minimal).
- `OcrException extends \RuntimeException`.

### B5. Test

One PHPUnit smoke test, no real engine/models:

- A tiny fake binary (a shell/php stub that echoes a canned JSON object) stands
  in for `arboocr_demo`. Point `Engine` at it via `binPath`.
- Assert: happy path parses into a `PageResult` with the right line count/text;
  a stub that exits non-zero / prints garbage raises `OcrException`.
- This exercises argv-building, `--json` handling, and the parse/error paths
  without any C++ or ONNX in CI.

---

## Build & sequencing

1. **A first.** Implement `--json`, add the release workflow, push a real tag,
   confirm the workflow produces `arboocr-windows-x64.zip` +
   `arboocr-linux-x64.tar.gz` and that each unzips to a folder whose
   `arboocr_demo --json --image <sample> --models-dir <models>` prints valid
   JSON on a clean machine (no repo, no vcpkg).
2. **B second, against the real artifact.** Build the PHP package pointing its
   pinned tag at A's release. Verify install → download → `recognize()` on the
   sample image end-to-end.

Building B against a *real* release (not assumptions about the artifact's
internal layout) is deliberate: the exact set of bundled DLLs/.so and the
folder structure are decided by A's packaging step, and B's `binPath` default
must match it.

## Risks / notes

- **vcpkg cold build time** in CI (OpenCV + onnxruntime from source) can be
  10s of minutes. Binary caching mitigates; acceptable for tag-triggered
  releases which are infrequent.
- **Linux .so bundling / RPATH** is the fiddliest part of A2 — the binary must
  locate its bundled onnxruntime/opencv at runtime via `$ORIGIN`. Verify on a
  container without the build deps installed.
- **onnxruntime license / third-party notices** — the release archive ships
  onnxruntime + opencv binaries; include `LICENSE` and `THIRD_PARTY_NOTICES.md`
  in the archive.
