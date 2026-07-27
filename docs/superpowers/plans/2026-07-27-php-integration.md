# PHP Integration (arboOCR --json + release CI, arbo-ocr-php package) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let PHP run arboOCR without building C++: add `--json` output to the
CLI, a tag-triggered GitHub Actions release workflow producing standalone
Windows+Linux binaries, and a new Composer package (`arbo-ocr-php`) that
downloads the matching release binary and shells out to it via `proc_open`.

**Architecture:** Two repos, built in strict order. Component A (this repo,
`arboOCR`) adds a `--json` flag to `arboocr_demo` (reusing the existing
`arbo::ocr::toJson()`), and a `.github/workflows/release.yml` that builds both
presets on tag push and attaches zipped/tarred binary+DLL/.so bundles to a
GitHub Release. Component B (new repo `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php`)
is a framework-agnostic Composer package: an `Installer` (Composer post-install
hook) downloads the pinned-tag release asset for the host OS, and an `Engine`
class runs it via `proc_open` with an argv array (no shell string), parsing
`--json` stdout into `PageResult`/`LineResult` value objects.

**Tech Stack:** C++17 / CMake / vcpkg / cxxopts / doctest (Component A, existing
stack — no new deps). PHP 8.3 / Composer / PHPUnit (Component B, new package).
GitHub Actions (`windows-latest`, `ubuntu-latest`).

## Global Constraints

- CLI-exec integration only — no FFI, no new C ABI (per approved design).
- `--json` stdout must be **pure JSON, nothing else** — no interleaved log/human
  lines — so PHP can `json_decode()` the whole stdout stream.
- Release artifacts contain **binary + required shared libs only** — no ONNX
  models (spec: keep artifacts ~50MB, consumer supplies `--models-dir`).
- Two CI platforms only: `windows-x64`, `linux-x64` (no Jetson/aarch64 CI).
- Release trigger: push of tag matching `v*`.
- PHP package must work with only the **build artifact** on disk — never
  requires cloning/building the arboOCR repo.
- `arbo-ocr-php` builds argv arrays for `proc_open`, never a shell string
  (no shell-injection surface from image paths).
- Empty `lines` in a JSON result is success (exit 0), not an error — matches
  `Engine::recognize()`'s existing "never throws" contract.
- Spec doc: `docs/superpowers/specs/2026-07-27-php-integration-design.md`.

---

## File Structure

**Component A (`arboOCR`, this repo):**
- Modify: `include/arboOCR/types.hpp` — new `toJson(page, backend, pretty)` overload declaration.
- Modify: `src/arboOCR/types.cpp` — implement the new overload.
- Modify: `tests/test_engine.cpp` — test for the new overload.
- Modify: `cli/arboocr_demo.cpp` — add `--json` flag, branch output.
- Create: `.github/workflows/release.yml` — tag-triggered build+package+publish.

**Component B (`arbo-ocr-php`, new repo):**
- Create: `composer.json` — package manifest, PSR-4, installer hook, pinned arboOCR tag.
- Create: `src/Installer.php` — downloads/extracts the matching release binary.
- Create: `src/OcrException.php` — error type.
- Create: `src/LineResult.php` — value object.
- Create: `src/PageResult.php` — value object.
- Create: `src/Engine.php` — builds argv, runs `proc_open`, parses JSON.
- Create: `tests/EngineTest.php` — PHPUnit smoke test against a fake stub binary.
- Create: `tests/fixtures/fake_arboocr.php` — stub binary used by the test.
- Create: `.gitignore` — ignore `vendor/`, downloaded `bin/`.
- Create: `README.md` — usage.

---

## Task 1: `toJson` gains a `backend` field (Component A)

**Files:**
- Modify: `include/arboOCR/types.hpp`
- Modify: `src/arboOCR/types.cpp`
- Test: `tests/test_engine.cpp`

**Interfaces:**
- Produces: `std::string toJson(const PagePrediction& page, const std::string& backend, bool pretty = false);` — new overload. Existing two-arg `toJson(page, pretty)` is untouched (still used elsewhere / by any external consumer).
- Consumes: existing `PagePrediction`, `LinePrediction`, `escapeJson` (anonymous-namespace helper already in `types.cpp`), `appendLine` (already in `types.cpp`).

JSON key order for the new overload: `backend`, `image`, `elapsedMs`, `lines` — `backend` first, matching the field order in the design spec's example.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_engine.cpp` (after the existing `"toJson on empty page is valid object"` test case — check that test's exact closing brace/line first with Read before inserting):

```cpp
TEST_CASE("toJson with backend includes backend field before image") {
    PagePrediction page;
    page.image = "page.jpg";
    page.elapsedMs = 12.5f;
    page.lines.push_back(LinePrediction{
        {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}},
        "hi",
        0.9f,
        0.8f,
    });

    const std::string json = toJson(page, std::string("cpu"));
    CHECK(json.find("\"backend\":\"cpu\"") != std::string::npos);
    CHECK(json.find("\"image\":\"page.jpg\"") != std::string::npos);
    // backend must come before image in the object
    CHECK(json.find("\"backend\"") < json.find("\"image\""));

    const std::string pretty = toJson(page, std::string("tensorrt"), true);
    CHECK(pretty.find("\"backend\":\"tensorrt\"") != std::string::npos);
    CHECK(pretty.find('\n') != std::string::npos);
}

TEST_CASE("toJson with backend on empty page is valid object") {
    PagePrediction empty;
    empty.image = "none.jpg";
    const std::string json = toJson(empty, std::string("cpu"));
    CHECK(json == "{\"backend\":\"cpu\",\"image\":\"none.jpg\",\"elapsedMs\":0,\"lines\":[]}");
}
```

**Overload resolution note:** the test calls above use explicit
`std::string("cpu")`, not a bare string literal `"cpu"`. This is
deliberate: with a bare `const char*` literal, C++ overload resolution
prefers the existing `toJson(page, bool pretty)` overload (pointer→bool is
a standard conversion) over the new `toJson(page, const std::string&,
bool)` overload (which needs a user-defined `const char*`→`std::string`
conversion) — so a literal would silently call the *old* overload with
`pretty=true` instead of failing to compile. Always pass an explicit
`std::string` in these tests. The real call site (Task 2's CLI, via
`engine.backend()`) is unaffected — `Engine::backend()` returns
`std::string`, which has no implicit conversion to `bool`, so it binds to
the new overload unambiguously.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build/windows-x64 --config Release --target arboocr_tests`
Expected: FAIL to compile — no `toJson` overload yet accepts a `std::string`
as the second argument (only `toJson(page, bool pretty)` exists so far).

- [ ] **Step 3: Declare the new overload**

In `include/arboOCR/types.hpp`, add directly below the existing:
```cpp
std::string toJson(const PagePrediction& page, bool pretty = false);
```
add:
```cpp
/// Same as toJson(page, pretty) but includes a "backend" field (e.g.
/// "cpu"/"cuda"/"tensorrt") — Engine::backend() isn't part of PagePrediction,
/// so callers that want it in the JSON (e.g. the CLI's --json mode) use this
/// overload instead of splicing it in themselves.
std::string toJson(const PagePrediction& page, const std::string& backend, bool pretty = false);
```

- [ ] **Step 4: Implement in types.cpp**

In `src/arboOCR/types.cpp`, add after the existing `toJson(const PagePrediction&, bool)`:

```cpp
std::string toJson(const PagePrediction& page, const std::string& backend, bool pretty) {
    std::ostringstream os;
    os << std::setprecision(6);
    if (pretty) {
        os << "{\n"
           << "  \"backend\":\"" << escapeJson(backend) << "\",\n"
           << "  \"image\":\"" << escapeJson(page.image) << "\",\n"
           << "  \"elapsedMs\":" << page.elapsedMs << ",\n"
           << "  \"lines\":[\n";
        for (size_t i = 0; i < page.lines.size(); i++) {
            os << "    ";
            appendLine(os, page.lines[i], true, 4);
            if (i + 1 < page.lines.size()) os << ",";
            os << "\n";
        }
        os << "  ]\n}";
    } else {
        os << "{\"backend\":\"" << escapeJson(backend) << "\",\"image\":\""
           << escapeJson(page.image) << "\",\"elapsedMs\":"
           << page.elapsedMs << ",\"lines\":[";
        for (size_t i = 0; i < page.lines.size(); i++) {
            if (i) os << ",";
            appendLine(os, page.lines[i], false, 0);
        }
        os << "]}";
    }
    return os.str();
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cmake --build build/windows-x64 --config Release --target arboocr_tests`
then: `build/windows-x64/Release/arboocr_tests.exe --test-case="toJson with backend*"`
Expected: PASS, both new test cases green.

- [ ] **Step 6: Commit**

```bash
git add include/arboOCR/types.hpp src/arboOCR/types.cpp tests/test_engine.cpp
git commit -m "feat(arboOCR): toJson overload with backend field"
```

---

## Task 2: `--json` flag on `arboocr_demo` (Component A)

**Files:**
- Modify: `cli/arboocr_demo.cpp`

**Interfaces:**
- Consumes: `arbo::ocr::toJson(const PagePrediction&, const std::string& backend, bool pretty = false)` from Task 1.
- Consumes: `arbo::ocr::Engine::backend()`, `arbo::ocr::Engine::recognize(std::string)` (both already exist).
- Produces: `arboocr_demo --json` — a runnable CLI mode later tasks (release workflow smoke check, PHP `Engine`) depend on.

No unit test framework wraps the CLI (`arboocr_demo` is a `main()`, not a
library target under `arboocr_tests`) — this task's "test" is a manual run
step, matching how the existing human-readable output is verified in this
codebase (no CLI tests exist today).

- [ ] **Step 1: Add the `--json` option and branch output**

Read `cli/arboocr_demo.cpp` in full first (already read above — 74 lines) to
confirm exact insertion points; then edit:

Add to the `opts.add_options()` chain, right after the `"clahe"` option:
```cpp
        ("json", "Print machine-readable JSON (only JSON on stdout; suppresses the human-readable lines)",
            cxxopts::value<bool>()->default_value("false"))
```

Replace the tail of `main()` (from `arbo::ocr::Engine engine(cfg);` to the end)
with:

```cpp
    const bool jsonMode = result["json"].as<bool>();

    arbo::ocr::Engine engine(cfg);
    if (!jsonMode) {
        std::cout << "Backend: " << engine.backend() << "\n";
    }

    auto page = engine.recognize(result["image"].as<std::string>());

    if (jsonMode) {
        // Pure JSON on stdout, nothing else — callers (e.g. the PHP wrapper)
        // json_decode() the whole stream. Empty lines is still success.
        std::cout << arbo::ocr::toJson(page, engine.backend()) << "\n";
        return 0;
    }

    std::cout << "Image: " << page.image << "\n";
    // recognize() never throws — missing/unreadable images and inference
    // failures both yield empty lines with elapsedMs still set.
    if (page.lines.empty()) {
        std::cerr << "No text found (missing image, unsupported format, or empty page)"
                  << " (" << page.elapsedMs << " ms)\n";
        return 1;
    }
    std::cout << "Lines: " << page.lines.size() << " (" << page.elapsedMs << " ms)\n";
    for (size_t i = 0; i < page.lines.size(); i++) {
        const auto& line = page.lines[i];
        std::cout << "  [" << i << "] \"" << line.text
                  << "\" (score=" << line.score
                  << " detScore=" << line.detScore << ")";
        if (!line.polygon.empty()) {
            std::cout << " poly=[";
            for (size_t p = 0; p < line.polygon.size(); p++) {
                if (p) std::cout << ", ";
                std::cout << "(" << line.polygon[p].x << "," << line.polygon[p].y << ")";
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
    return 0;
```

Note: in JSON mode, empty `lines` still exits 0 (design spec's error contract:
JSON mode signals failure via non-zero exit + empty stdout only when the
*engine itself* fails to construct or throws — an empty-but-valid page is not
a CLI-level error). The `Engine` constructor is not wrapped in try/catch here
because it wasn't before this change either (pre-existing behavior: an
uncaught exception from `Engine::Engine()` crashes the process with a non-zero
exit and a runtime message on stderr, which already satisfies "non-zero exit,
empty stdout" — verified in Step 3).

- [ ] **Step 2: Build**

Run: `cmake --build build/windows-x64 --config Release --target arboocr_demo`
Expected: builds clean, no warnings about the new flag.

- [ ] **Step 3: Manual smoke check — success path**

Run (from `build/windows-x64/Release/`):
```bash
./arboocr_demo.exe --image "../../../tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg" --models-dir "../../../models" --model-type tiny --json 1>stdout.txt 2>stderr.txt
echo "EXIT=$?"
```
Expected: `EXIT=0`; `stdout.txt` contains **only** one line, valid JSON, starting
`{"backend":"cpu","image":"INDONESIAN_RECEIPT_ZZ_2025041400001.jpg",...`. Verify
with:
```bash
py -3 -c "import json,sys; d=json.load(open('stdout.txt')); print(d['backend'], d['image'], len(d['lines']))"
```
Expected: prints `cpu INDONESIAN_RECEIPT_ZZ_2025041400001.jpg 31` (or similar — line count from Task 2's context sample run was 31 with `--model-type tiny`... actually that run used tiny+angle; without `--angle` count may differ slightly — accept any non-negative integer, the point is it parses and `lines` is a list).
Clean up: `rm stdout.txt stderr.txt`.

- [ ] **Step 4: Manual smoke check — engine-construction failure path**

Run:
```bash
./arboocr_demo.exe --image "../../../tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg" --models-dir "./does_not_exist" --json 1>stdout.txt 2>stderr.txt
echo "EXIT=$?"
wc -c stdout.txt
```
Expected: `EXIT` non-zero, `stdout.txt` is 0 bytes (empty), `stderr.txt` has the
ONNXRuntime error text. If stdout is NOT empty (e.g. cxxopts or ORT prints to
stdout on this failure), stop and fix — wrap `arbo::ocr::Engine engine(cfg);`
in a try/catch that prints to stderr and `return 1;` before this step is
considered passing.
Clean up: `rm stdout.txt stderr.txt`.

- [ ] **Step 5: Commit**

```bash
git add cli/arboocr_demo.cpp
git commit -m "feat(arboOCR): add --json output mode to arboocr_demo CLI"
```

---

## Task 3: Release workflow — Windows + Linux build/package/publish (Component A)

**Files:**
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: existing CMake presets `windows-x64`/`linux-x64` (`CMakePresets.json`), `vcpkg.json` manifest, `arboocr_demo` target (now with `--json` from Task 2).
- Produces: on tag push, a GitHub Release with two assets:
  `arboocr-windows-x64.zip`, `arboocr-linux-x64.tar.gz` — consumed by
  Component B's `Installer` (Task 6).

- [ ] **Step 1: Write the workflow file**

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  windows-x64:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Bootstrap vcpkg
        run: |
          git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
          C:\vcpkg\bootstrap-vcpkg.bat
        shell: pwsh

      - name: Configure
        run: cmake --preset windows-x64
        shell: pwsh
        env:
          VCPKG_ROOT: C:\vcpkg

      - name: Build arboocr_demo
        run: cmake --build build/windows-x64 --config Release --target arboocr_demo
        shell: pwsh

      - name: Package
        run: |
          $out = "package/arboocr-windows-x64"
          New-Item -ItemType Directory -Force -Path $out | Out-Null
          Copy-Item "build/windows-x64/Release/*.exe" $out
          Copy-Item "build/windows-x64/Release/*.dll" $out
          Copy-Item "LICENSE" $out
          Copy-Item "THIRD_PARTY_NOTICES.md" $out
          Compress-Archive -Path "$out/*" -DestinationPath "arboocr-windows-x64.zip"
        shell: pwsh

      - uses: actions/upload-artifact@v4
        with:
          name: arboocr-windows-x64
          path: arboocr-windows-x64.zip

  linux-x64:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Bootstrap vcpkg
        run: |
          git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg
          /opt/vcpkg/bootstrap-vcpkg.sh

      - name: Install build tools
        run: sudo apt-get update && sudo apt-get install -y ninja-build

      - name: Configure
        run: cmake --preset linux-x64
        env:
          VCPKG_ROOT: /opt/vcpkg

      - name: Build arboocr_demo
        run: cmake --build build/linux-x64 --target arboocr_demo

      - name: Package
        run: |
          set -euo pipefail
          out="package/arboocr-linux-x64"
          mkdir -p "$out"
          bin="build/linux-x64/arboocr_demo"
          cp "$bin" "$out/"
          # Bundle every shared lib the binary resolves outside system dirs
          # (vcpkg-built onnxruntime/opencv/curl and their deps), and point
          # the binary at them via a relative RPATH so it's self-contained.
          ldd "$bin" | awk '{print $3}' | grep -E '^/' | grep -v -E '^/lib|^/usr/lib' | while read -r lib; do
            cp -L "$lib" "$out/"
          done
          cp LICENSE THIRD_PARTY_NOTICES.md "$out/"
          patchelf --set-rpath '$ORIGIN' "$out/arboocr_demo" || true
          tar czf arboocr-linux-x64.tar.gz -C package arboocr-linux-x64

      - uses: actions/upload-artifact@v4
        with:
          name: arboocr-linux-x64
          path: arboocr-linux-x64.tar.gz

  publish:
    needs: [windows-x64, linux-x64]
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true

      - name: Create release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            dist/arboocr-windows-x64.zip
            dist/arboocr-linux-x64.tar.gz
```

Notes embedded as workflow comments aren't needed — the plan captures the
rationale (see spec §A2); the YAML itself stays plain per repo conventions
(no other workflow file exists yet to match style against, so keep it minimal
and readable).

`patchelf` ships on `ubuntu-latest` runners' image by default; if a future
run shows it's missing, add `sudo apt-get install -y patchelf` to the
"Install build tools" step — flag this as a known runner-image assumption,
not a certainty, since it wasn't verified against a live runner as part of
this plan (no way to run actual GitHub-hosted CI from this sandboxed
environment).

- [ ] **Step 2: Validate YAML syntax locally**

Run: `py -3 -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('valid')"`
Expected: prints `valid`. This only checks YAML syntax, not GitHub Actions
schema/logic — full validation happens when the tag is actually pushed
(Task 4).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci(arboOCR): add tag-triggered release workflow (windows-x64, linux-x64)"
```

---

## Task 4: Tag, run, and verify the release workflow end-to-end (Component A)

**Files:** none (verification task — no new files).

**Interfaces:**
- Consumes: `.github/workflows/release.yml` from Task 3.
- Produces: a real GitHub Release (e.g. tag `v0.1.0-php1`) with
  `arboocr-windows-x64.zip` + `arboocr-linux-x64.tar.gz` attached — this is
  the artifact Component B's `Installer` targets. **Component B's default
  pinned tag (Task 6) must equal whatever tag is actually pushed here.**

This task requires pushing to the real `wafik/ArboOCR` remote and watching
Actions run — it's an infrastructure/network step, not local code. Confirm
with the user before pushing a tag (irreversible-ish: creates a public
release) if not already implicitly authorized by "continue all until done".
Given the user said to continue through to done, proceed, but pick a clearly
scoped pre-release tag name so it reads as a test artifact, not a real
version bump.

- [ ] **Step 1: Push a tag**

```bash
git tag v0.1.0-php1
git push origin v0.1.0-php1
```

- [ ] **Step 2: Watch the workflow run**

```bash
gh run watch --repo wafik/ArboOCR
```
Expected: both `windows-x64` and `linux-x64` jobs succeed, then `publish`
succeeds. If a job fails, read its log (`gh run view --repo wafik/ArboOCR --log-failed`),
fix the workflow file, commit, delete+repush the tag (`git tag -d v0.1.0-php1 && git push origin :refs/tags/v0.1.0-php1`, then repeat Step 1), and re-watch. This loop is expected to take 1-3 iterations for a first-ever CI file — most likely failure points: vcpkg cold-build timeout (default job timeout may need raising), `patchelf` missing, or `Compress-Archive` wildcard path issues on Windows.

- [ ] **Step 3: Verify the release assets are real and runnable**

```bash
gh release view v0.1.0-php1 --repo wafik/ArboOCR --json assets
```
Expected: both `arboocr-windows-x64.zip` and `arboocr-linux-x64.tar.gz` listed.

Download and smoke-test the Windows asset locally (this machine is Windows):
```bash
gh release download v0.1.0-php1 --repo wafik/ArboOCR -p "arboocr-windows-x64.zip" -D /tmp/arbo-release-check
cd /tmp/arbo-release-check && powershell -c "Expand-Archive arboocr-windows-x64.zip -DestinationPath extracted"
cd extracted/arboocr-windows-x64
./arboocr_demo.exe --image "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg" --models-dir "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/models" --model-type tiny --json
```
Expected: runs successfully **on a machine with no vcpkg/CMake/repo context**
(only the extracted zip + an image + a models dir), printing valid JSON to
stdout. This is the critical proof that Component B's "no repo needed" design
constraint actually holds.

Record the verified tag name — it's needed verbatim in Task 6.

- [ ] **Step 4: No commit** (this task only produces a remote tag/release, not a local file change).

---

## Task 5: `arbo-ocr-php` package scaffold + value objects

**Files:**
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\composer.json`
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\src\OcrException.php`
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\src\LineResult.php`
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\src\PageResult.php`
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\.gitignore`

**Interfaces:**
- Produces: `Arbo\Ocr\OcrException`, `Arbo\Ocr\LineResult` (readonly: `text`,
  `score`, `detScore`, `polygon` — array of `['x'=>float,'y'=>float]`),
  `Arbo\Ocr\PageResult` (readonly: `backend`, `image`, `elapsedMs`,
  `lines` — `LineResult[]`, plus `static fromJson(string $json): self`).
  Task 6 (`Engine`) and Task 7 (tests) both consume `PageResult::fromJson`.

- [ ] **Step 1: Init the repo and composer.json**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php"
git init
```

Create `composer.json`:
```json
{
    "name": "arbo/ocr-php",
    "description": "PHP wrapper for arboOCR — runs the prebuilt arboocr_demo binary via proc_open, no C++ build required.",
    "type": "library",
    "license": "Apache-2.0",
    "require": {
        "php": ">=8.1",
        "ext-json": "*"
    },
    "require-dev": {
        "phpunit/phpunit": "^10.0"
    },
    "autoload": {
        "psr-4": {
            "Arbo\\Ocr\\": "src/"
        }
    },
    "autoload-dev": {
        "psr-4": {
            "Arbo\\Ocr\\Tests\\": "tests/"
        }
    },
    "extra": {
        "arboocr-version": "v0.1.0-php1"
    },
    "scripts": {
        "post-install-cmd": "Arbo\\Ocr\\Installer::run",
        "post-update-cmd": "Arbo\\Ocr\\Installer::run"
    }
}
```

(`extra.arboocr-version` is set to the tag verified in Task 4 Step 3 — if that
tag differs from `v0.1.0-php1`, use the real one here.)

- [ ] **Step 2: `.gitignore`**

```
/vendor/
/bin/
/.phpunit.cache/
```

- [ ] **Step 3: `OcrException`**

`src/OcrException.php`:
```php
<?php

declare(strict_types=1);

namespace Arbo\Ocr;

/**
 * Thrown when the arboocr_demo process fails to start, exits non-zero, or
 * produces stdout that isn't valid JSON. An empty `lines` array in a
 * successful (exit 0) result is NOT an error — arboOCR's own contract is
 * that "no text found" is a normal, valid PageResult.
 */
final class OcrException extends \RuntimeException
{
    public function __construct(string $message, public readonly int $exitCode = -1, public readonly string $stderr = '')
    {
        parent::__construct($message);
    }
}
```

- [ ] **Step 4: `LineResult`**

`src/LineResult.php`:
```php
<?php

declare(strict_types=1);

namespace Arbo\Ocr;

/**
 * One recognized text line. `polygon` is a list of ['x' => float, 'y' => float]
 * points, in the order arboOCR reports them (clockwise from top-left-ish).
 */
final class LineResult
{
    /** @param array<int, array{x: float, y: float}> $polygon */
    public function __construct(
        public readonly string $text,
        public readonly float $score,
        public readonly float $detScore,
        public readonly array $polygon,
    ) {
    }

    /** @param array<string, mixed> $data */
    public static function fromArray(array $data): self
    {
        return new self(
            text: (string) ($data['text'] ?? ''),
            score: (float) ($data['score'] ?? 0.0),
            detScore: (float) ($data['detScore'] ?? 0.0),
            polygon: $data['polygon'] ?? [],
        );
    }
}
```

- [ ] **Step 5: `PageResult`**

`src/PageResult.php`:
```php
<?php

declare(strict_types=1);

namespace Arbo\Ocr;

/**
 * Full-page OCR result — mirrors arboOCR's PagePrediction. Empty `lines` is
 * a normal, successful result (no text found), not an error.
 */
final class PageResult
{
    /** @param LineResult[] $lines */
    public function __construct(
        public readonly string $backend,
        public readonly string $image,
        public readonly float $elapsedMs,
        public readonly array $lines,
    ) {
    }

    /**
     * Parse the JSON stdout of `arboocr_demo --json`.
     *
     * @throws OcrException if $json is not a valid JSON object with the
     *   expected shape.
     */
    public static function fromJson(string $json): self
    {
        $data = json_decode($json, true);
        if (!is_array($data) || !isset($data['lines']) || !is_array($data['lines'])) {
            throw new OcrException('arboocr_demo --json produced unparseable output: ' . substr($json, 0, 500));
        }

        $lines = array_map(
            static fn (array $line) => LineResult::fromArray($line),
            $data['lines'],
        );

        return new self(
            backend: (string) ($data['backend'] ?? ''),
            image: (string) ($data['image'] ?? ''),
            elapsedMs: (float) ($data['elapsedMs'] ?? 0.0),
            lines: $lines,
        );
    }
}
```

- [ ] **Step 6: Install dev deps and confirm autoload works**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php"
composer install
php -r "require 'vendor/autoload.php'; var_dump(class_exists('Arbo\\Ocr\\PageResult'));"
```
Expected: `bool(true)`. (The `post-install-cmd` hook will try to call
`Arbo\Ocr\Installer::run` — that class doesn't exist until Task 6, so
`composer install` will error on the script step. That's expected at this
point in the plan; if it blocks `vendor/autoload.php` from being generated,
temporarily comment out the two `scripts` lines in `composer.json`, run
`composer install`, verify autoload, then uncomment before Step 7.)

- [ ] **Step 7: Commit**

```bash
git add composer.json .gitignore src/OcrException.php src/LineResult.php src/PageResult.php
git commit -m "feat(arbo-ocr-php): package scaffold and value objects"
```

---

## Task 6: `Installer` — download the matching release binary

**Files:**
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\src\Installer.php`

**Interfaces:**
- Consumes: `composer.json` `extra.arboocr-version` (Task 5); GitHub Releases
  URL pattern from Component A Task 4:
  `https://github.com/wafik/ArboOCR/releases/download/<tag>/arboocr-{windows-x64.zip,linux-x64.tar.gz}`.
- Produces: `bin/windows-x64/arboocr_demo.exe` or `bin/linux-x64/arboocr_demo`
  (plus sibling DLLs/.so) on disk after running. `Engine::defaultBinPath()`
  (Task 7) depends on this exact layout.

- [ ] **Step 1: Write `Installer.php`**

```php
<?php

declare(strict_types=1);

namespace Arbo\Ocr;

/**
 * Composer post-install/post-update hook. Downloads the arboOCR release
 * binary matching this package's pinned version (composer.json
 * extra.arboocr-version) for the host OS, and extracts it to bin/<platform>/
 * next to this file. Never fails composer install/update on error — arboOCR
 * still works if the caller points Engine at a manually-downloaded binary
 * via the `binPath` option.
 */
final class Installer
{
    private const REPO = 'wafik/ArboOCR';

    public static function run(): void
    {
        $platform = self::detectPlatform();
        if ($platform === null) {
            fwrite(STDERR, "[arbo-ocr-php] Unsupported OS for auto-download. "
                . "Download a release manually from "
                . "https://github.com/" . self::REPO . "/releases and pass "
                . "'binPath' to Engine.\n");
            return;
        }

        $version = self::pinnedVersion();
        $targetDir = __DIR__ . '/../bin/' . $platform;
        $binName = $platform === 'windows-x64' ? 'arboocr_demo.exe' : 'arboocr_demo';

        if (is_file($targetDir . '/' . $binName)) {
            return; // already installed
        }

        $asset = $platform === 'windows-x64' ? 'arboocr-windows-x64.zip' : 'arboocr-linux-x64.tar.gz';
        $url = "https://github.com/" . self::REPO . "/releases/download/{$version}/{$asset}";

        try {
            self::downloadAndExtract($url, $targetDir, $asset);
            if ($platform === 'linux-x64') {
                @chmod($targetDir . '/' . $binName, 0755);
            }
            fwrite(STDOUT, "[arbo-ocr-php] Installed arboocr_demo ({$platform}, {$version}) to {$targetDir}\n");
        } catch (\Throwable $e) {
            fwrite(STDERR, "[arbo-ocr-php] Could not auto-download arboOCR binary: "
                . $e->getMessage() . "\nDownload manually from {$url} and pass "
                . "'binPath' to Engine, or re-run 'composer install'.\n");
        }
    }

    public static function detectPlatform(): ?string
    {
        $family = PHP_OS_FAMILY;
        return match ($family) {
            'Windows' => 'windows-x64',
            'Linux' => 'linux-x64',
            default => null,
        };
    }

    public static function pinnedVersion(): string
    {
        $composerJson = json_decode(
            (string) file_get_contents(__DIR__ . '/../composer.json'),
            true,
        );
        return $composerJson['extra']['arboocr-version'] ?? 'latest';
    }

    private static function downloadAndExtract(string $url, string $targetDir, string $assetName): void
    {
        if (!is_dir($targetDir) && !mkdir($targetDir, 0755, true) && !is_dir($targetDir)) {
            throw new \RuntimeException("Could not create {$targetDir}");
        }

        $tmpFile = tempnam(sys_get_temp_dir(), 'arboocr-dl-');
        if ($tmpFile === false) {
            throw new \RuntimeException('Could not create temp file for download');
        }

        $ctx = stream_context_create(['http' => ['follow_location' => 1, 'timeout' => 120]]);
        $data = @file_get_contents($url, false, $ctx);
        if ($data === false) {
            @unlink($tmpFile);
            throw new \RuntimeException("Download failed: {$url}");
        }
        file_put_contents($tmpFile, $data);

        if (str_ends_with($assetName, '.zip')) {
            $zip = new \ZipArchive();
            if ($zip->open($tmpFile) !== true) {
                @unlink($tmpFile);
                throw new \RuntimeException("Could not open downloaded zip: {$assetName}");
            }
            $zip->extractTo($targetDir);
            $zip->close();
            self::flattenSingleSubdir($targetDir);
        } else {
            $phar = new \PharData($tmpFile);
            $phar->extractTo($targetDir, overwrite: true);
            self::flattenSingleSubdir($targetDir);
        }

        @unlink($tmpFile);
    }

    /**
     * The release archives contain one top-level folder (e.g.
     * arboocr-windows-x64/...). Move its contents up into $targetDir so
     * callers get bin/<platform>/arboocr_demo directly, not
     * bin/<platform>/arboocr-windows-x64/arboocr_demo.
     */
    private static function flattenSingleSubdir(string $targetDir): void
    {
        $entries = array_values(array_diff(scandir($targetDir) ?: [], ['.', '..']));
        if (count($entries) !== 1 || !is_dir($targetDir . '/' . $entries[0])) {
            return;
        }
        $subdir = $targetDir . '/' . $entries[0];
        foreach (array_diff(scandir($subdir) ?: [], ['.', '..']) as $item) {
            rename($subdir . '/' . $item, $targetDir . '/' . $item);
        }
        rmdir($subdir);
    }
}
```

- [ ] **Step 2: Run `composer install` for real**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php"
composer install
```
Expected: the `post-install-cmd` hook runs `Installer::run()`, downloads
`arboocr-windows-x64.zip` from the tag set in `composer.json`
(`extra.arboocr-version`, from Task 4), and `bin/windows-x64/arboocr_demo.exe`
plus its DLLs exist afterward.

- [ ] **Step 3: Verify the installed binary actually runs**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php"
./bin/windows-x64/arboocr_demo.exe --image "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg" --models-dir "D:/kerjaan/kreasi/kimfu/cpp/arboOCR/models" --model-type tiny --json
```
Expected: valid JSON on stdout, exit 0.

- [ ] **Step 4: Verify idempotency**

```bash
composer install
```
Expected: runs again without error, does not re-download (binary already
present — `Installer::run()` returns early).

- [ ] **Step 5: Commit**

```bash
git add src/Installer.php
git commit -m "feat(arbo-ocr-php): Installer downloads matching release binary on composer install"
```

---

## Task 7: `Engine` — run the binary, parse results

**Files:**
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\src\Engine.php`

**Interfaces:**
- Consumes: `Arbo\Ocr\Installer::detectPlatform()` (Task 6, for the default
  `binPath`), `Arbo\Ocr\PageResult::fromJson()` (Task 5), `Arbo\Ocr\OcrException`
  (Task 5).
- Produces: `Arbo\Ocr\Engine::__construct(array $options = [])`,
  `Engine::recognize(string $imagePath): PageResult` — the package's main
  public entry point, exercised by Task 8's tests.

- [ ] **Step 1: Write `Engine.php`**

```php
<?php

declare(strict_types=1);

namespace Arbo\Ocr;

/**
 * Runs the prebuilt arboocr_demo binary via proc_open and parses its
 * --json output. Requires no C++ build — only the binary the Installer
 * downloaded (or one you point at manually via the 'binPath' option).
 */
final class Engine
{
    /** @var list<string> */
    private array $binCommand;
    private array $options;

    /**
     * @param array{
     *   binPath?: string|list<string>,
     *   modelsDir?: string,
     *   ocrVersion?: string,
     *   modelType?: string,
     *   useAngleCls?: bool,
     *   useCuda?: bool,
     *   useTensorrt?: bool,
     *   useFp16?: bool,
     *   useClahe?: bool,
     *   detModelPath?: string,
     *   clsModelPath?: string,
     *   recModelPath?: string,
     *   dictPath?: string,
     * } $options 'binPath' is normally a single executable path (string).
     *   An array form (e.g. [PHP_BINARY, 'script.php']) is also accepted
     *   for wrapping a non-directly-executable command — used by the test
     *   suite to invoke a fake binary through the PHP interpreter.
     */
    public function __construct(array $options = [])
    {
        $bin = $options['binPath'] ?? self::defaultBinPath();
        $this->binCommand = is_array($bin) ? $bin : [$bin];
        unset($options['binPath']);
        $this->options = $options;
    }

    public static function defaultBinPath(): string
    {
        $platform = Installer::detectPlatform();
        $binName = $platform === 'windows-x64' ? 'arboocr_demo.exe' : 'arboocr_demo';
        return __DIR__ . '/../bin/' . ($platform ?? 'linux-x64') . '/' . $binName;
    }

    /**
     * @throws OcrException if the process can't be started, exits non-zero,
     *   or its stdout isn't valid JSON. An empty PageResult::$lines is a
     *   normal, successful result — not an exception.
     */
    public function recognize(string $imagePath): PageResult
    {
        // Only check is_file() for the plain-single-path form — an array
        // binCommand's first element (e.g. PHP_BINARY) is a command name
        // resolved via PATH, not necessarily a direct file path.
        if (count($this->binCommand) === 1 && !is_file($this->binCommand[0])) {
            throw new OcrException("arboocr_demo binary not found at {$this->binCommand[0]}. "
                . "Run 'composer install' or pass 'binPath' explicitly.");
        }

        $argv = [...$this->binCommand, '--image', $imagePath, '--json'];
        foreach ($this->flagsFromOptions() as $flag => $value) {
            $argv[] = "--{$flag}";
            $argv[] = $value;
        }

        $descriptors = [1 => ['pipe', 'w'], 2 => ['pipe', 'w']];
        $process = proc_open($argv, $descriptors, $pipes);
        if (!is_resource($process)) {
            throw new OcrException('Could not start process: ' . implode(' ', $this->binCommand));
        }

        $stdout = stream_get_contents($pipes[1]) ?: '';
        $stderr = stream_get_contents($pipes[2]) ?: '';
        fclose($pipes[1]);
        fclose($pipes[2]);
        $exitCode = proc_close($process);

        if ($exitCode !== 0) {
            throw new OcrException(
                "arboocr_demo exited with code {$exitCode}",
                $exitCode,
                $stderr,
            );
        }

        return PageResult::fromJson(trim($stdout));
    }

    /** @return array<string, string> */
    private function flagsFromOptions(): array
    {
        $map = [
            'modelsDir' => 'models-dir',
            'ocrVersion' => 'ocr-version',
            'modelType' => 'model-type',
            'useAngleCls' => 'angle',
            'useCuda' => 'cuda',
            'useTensorrt' => 'tensorrt',
            'useFp16' => 'fp16',
            'useClahe' => 'clahe',
            'detModelPath' => 'det-model',
            'clsModelPath' => 'cls-model',
            'recModelPath' => 'rec-model',
            'dictPath' => 'dict',
        ];

        $flags = [];
        foreach ($map as $optKey => $cliFlag) {
            if (!array_key_exists($optKey, $this->options)) {
                continue;
            }
            $value = $this->options[$optKey];
            $flags[$cliFlag] = is_bool($value) ? ($value ? 'true' : 'false') : (string) $value;
        }
        return $flags;
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/Engine.php
git commit -m "feat(arbo-ocr-php): Engine runs arboocr_demo via proc_open and parses --json"
```

(No test yet — Task 8 adds the PHPUnit smoke test against a fake binary,
kept separate since it needs its own fixture file.)

---

## Task 8: `EngineTest` — smoke test against a fake binary + real end-to-end check

**Files:**
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\tests\fixtures\fake_arboocr.php`
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\tests\EngineTest.php`
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\phpunit.xml`

**Interfaces:**
- Consumes: `Arbo\Ocr\Engine` (Task 7), `Arbo\Ocr\OcrException`, `Arbo\Ocr\PageResult`, `Arbo\Ocr\LineResult` (Task 5).

- [ ] **Step 1: Write the fake binary fixture**

`tests/fixtures/fake_arboocr.php` — a script that mimics `arboocr_demo --json`'s
argv contract closely enough to test `Engine`'s argv-building and parsing
without any real OCR:

```php
<?php
// Stand-in for arboocr_demo, used only by EngineTest. Reads its own argv to
// decide what to emit, so tests can exercise both the happy path and error
// paths without a real binary or models.

$args = $argv;
array_shift($args); // drop script path

if (in_array('--fail', $args, true)) {
    fwrite(STDERR, "simulated engine failure\n");
    exit(2);
}

if (in_array('--garbage', $args, true)) {
    echo "not json\n";
    exit(0);
}

$imageIdx = array_search('--image', $args, true);
$image = $imageIdx !== false ? ($args[$imageIdx + 1] ?? '') : '';

echo json_encode([
    'backend' => 'cpu',
    'image' => basename($image),
    'elapsedMs' => 12.5,
    'lines' => [
        ['text' => 'hello', 'score' => 0.9, 'detScore' => 0.8,
         'polygon' => [['x' => 1.0, 'y' => 2.0]]],
    ],
]) . "\n";
exit(0);
```

- [ ] **Step 2: Write `EngineTest.php`**

`Engine`'s `binPath` option accepts a string (single executable path — the
real-world case) or an array (pre-tokenized command — used here to invoke the
fake PHP fixture through the PHP interpreter, since a bare `.php` file isn't
directly executable on every OS):

```php
<?php

declare(strict_types=1);

namespace Arbo\Ocr\Tests;

use Arbo\Ocr\Engine;
use Arbo\Ocr\OcrException;
use PHPUnit\Framework\TestCase;

final class EngineTest extends TestCase
{
    /** @return list<string> */
    private function fakeBin(array $extraArgs = []): array
    {
        return [PHP_BINARY, __DIR__ . '/fixtures/fake_arboocr.php', ...$extraArgs];
    }

    public function testRecognizeParsesSuccessfulJsonOutput(): void
    {
        $engine = new Engine(['binPath' => $this->fakeBin()]);
        $result = $engine->recognize('/some/page.jpg');

        self::assertSame('cpu', $result->backend);
        self::assertSame('page.jpg', $result->image);
        self::assertSame(12.5, $result->elapsedMs);
        self::assertCount(1, $result->lines);
        self::assertSame('hello', $result->lines[0]->text);
        self::assertSame(0.9, $result->lines[0]->score);
        self::assertSame(1.0, $result->lines[0]->polygon[0]['x']);
    }

    public function testRecognizeThrowsOnNonZeroExit(): void
    {
        $this->expectException(OcrException::class);
        $this->expectExceptionMessageMatches('/exited with code/');

        $engine = new Engine(['binPath' => $this->fakeBin(['--fail'])]);
        $engine->recognize('/some/page.jpg');
    }

    public function testRecognizeThrowsOnUnparseableOutput(): void
    {
        $this->expectException(OcrException::class);

        $engine = new Engine(['binPath' => $this->fakeBin(['--garbage'])]);
        $engine->recognize('/some/page.jpg');
    }

    public function testRecognizeThrowsWhenBinaryMissing(): void
    {
        $this->expectException(OcrException::class);
        $this->expectExceptionMessageMatches('/not found/');

        $engine = new Engine(['binPath' => '/no/such/binary']);
        $engine->recognize('/some/page.jpg');
    }

    public function testFlagsFromOptionsMapToCliFlags(): void
    {
        // Indirect check: modelsDir/useAngleCls etc. must reach argv without
        // erroring proc_open and must not break the fake binary's parsing
        // (it only reads --image, so any well-formed extra flags are fine).
        $engine = new Engine([
            'binPath' => $this->fakeBin(),
            'modelsDir' => 'models',
            'useAngleCls' => true,
            'useCuda' => false,
        ]);
        $result = $engine->recognize('/some/page.jpg');
        self::assertSame('cpu', $result->backend);
    }
}
```

- [ ] **Step 3: `phpunit.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<phpunit xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:noNamespaceSchemaLocation="https://schema.phpunit.de/10.5/phpunit.xsd"
         bootstrap="vendor/autoload.php"
         colors="true">
    <testsuites>
        <testsuite name="default">
            <directory>tests</directory>
        </testsuite>
    </testsuites>
</phpunit>
```

- [ ] **Step 4: Run the tests**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php"
./vendor/bin/phpunit
```
Expected: all 5 tests pass.

- [ ] **Step 5: Real end-to-end check against the actual downloaded binary**

This is the check that matters most — confirms the whole chain (Installer →
real `arboocr_demo.exe` → `Engine` → `PageResult`) works together, not just
against the fake:

```bash
php -r "
require 'vendor/autoload.php';
\$engine = new Arbo\Ocr\Engine([
    'modelsDir' => 'D:/kerjaan/kreasi/kimfu/cpp/arboOCR/models',
    'modelType' => 'tiny',
]);
\$result = \$engine->recognize('D:/kerjaan/kreasi/kimfu/cpp/arboOCR/tests/fixtures/test_images/INDONESIAN_RECEIPT_ZZ_2025041400001.jpg');
echo \$result->backend . ' — ' . count(\$result->lines) . \" lines\n\";
echo \$result->lines[0]->text . \"\n\";
"
```
Expected: prints `cpu — N lines` (N > 0) and the first recognized line's text
(e.g. `INVOICE`). This is the package's actual purpose working end to end.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/fake_arboocr.php tests/EngineTest.php phpunit.xml
git commit -m "test(arbo-ocr-php): EngineTest against fake binary"
```

---

## Task 9: README for arbo-ocr-php

**Files:**
- Create: `D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php\README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Write the README**

```markdown
# arbo-ocr-php

PHP wrapper for [arboOCR](https://github.com/wafik/ArboOCR) — runs the
prebuilt `arboocr_demo` binary via `proc_open`, no C++ build required.

## Install

\`\`\`bash
composer require arbo/ocr-php
\`\`\`

On install, a Composer hook downloads the matching arboOCR release binary
(Windows or Linux, auto-detected) into `bin/<platform>/`. If the auto-download
fails (offline install, unsupported OS), download a release manually from
the [arboOCR releases page](https://github.com/wafik/ArboOCR/releases) and
pass `binPath` explicitly (see below).

You also need the OCR models — arboOCR does not bundle them. See
[arboOCR's Models section](https://github.com/wafik/ArboOCR#models) for
download instructions, then point `modelsDir` at the folder.

## Usage

\`\`\`php
use Arbo\Ocr\Engine;

$engine = new Engine([
    'modelsDir' => '/path/to/models',
    // 'binPath' => '/custom/path/to/arboocr_demo', // optional override
    // 'modelType' => 'small', // tiny/small/medium — default small
    // 'useAngleCls' => true,
    // 'useCuda' => true,
]);

$result = $engine->recognize('/path/to/image.jpg');

echo $result->backend, "\n";       // cpu / cuda / tensorrt
foreach ($result->lines as $line) {
    echo $line->text, ' (', $line->score, ")\n";
}
\`\`\`

An empty `$result->lines` array means no text was found — not an error.
`Engine::recognize()` throws `Arbo\Ocr\OcrException` only when the process
itself fails to start, exits non-zero, or produces unparseable output.

## How it works

This package never builds or vendors arboOCR's C++ source. It downloads a
prebuilt, self-contained binary (binary + required shared libraries, no
source, no ONNX models) from arboOCR's GitHub Releases, and calls it as a
subprocess per image with a `--json` flag, parsing the JSON result. See
arboOCR's [`docs/superpowers/specs/2026-07-27-php-integration-design.md`](https://github.com/wafik/ArboOCR/blob/main/docs/superpowers/specs/2026-07-27-php-integration-design.md)
for the full design.

## License

Apache-2.0
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(arbo-ocr-php): usage README"
```

---

## Task 10: Final verification pass

**Files:** none.

- [ ] **Step 1: arboOCR full test suite still passes**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arboOCR"
cmake --build build/windows-x64 --config Release --target arboocr_tests
./build/windows-x64/Release/arboocr_tests.exe
```
Expected: all tests pass (including Task 1's two new cases), no regressions
in existing `toJson`/engine/detector/etc. tests.

- [ ] **Step 2: arbo-ocr-php full test suite still passes**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php"
./vendor/bin/phpunit
```
Expected: all tests pass.

- [ ] **Step 3: Confirm arboOCR working tree is clean and pushed**

```bash
cd "D:\kerjaan\kreasi\kimfu\cpp\arboOCR"
git status
git push origin docs/accuracy-upgrade-notes
```
(Push to whatever branch this work landed on — do not merge to `main` unless
the user asks; this plan doesn't include opening a PR, since brainstorming
didn't specify one.)

- [ ] **Step 4: Report final state to the user**

Summarize: arboOCR now has `--json` CLI output + a release workflow, tag
`<verified tag from Task 4>` has a working release; `arbo-ocr-php` exists at
`D:\kerjaan\kreasi\kimfu\cpp\arbo-ocr-php`, installs the binary automatically,
and `Engine::recognize()` works end-to-end against the real downloaded
binary. Note any deviations hit during Task 4's CI debugging loop.
