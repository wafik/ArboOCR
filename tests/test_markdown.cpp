#include <string>

#include <doctest/doctest.h>
#include "arboOCR/markdown.hpp"

using namespace arbo::ocr;

namespace {

// Axis-aligned line box in detector order (TL, TR, BR, BL) — the same shape
// sortLinesReadingOrder and spanToPolygon assume.
LinePrediction makeLine(const std::string& text, float x0, float y0, float x1, float y1) {
    LinePrediction line;
    line.polygon = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    line.text = text;
    line.score = 0.9f;
    line.detScore = 0.9f;
    return line;
}

WordBox makeWord(const std::string& text, float x0, float y0, float x1, float y1) {
    WordBox word;
    word.polygon = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    word.text = text;
    word.score = 0.9f;
    return word;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("toMarkdown on an empty page returns an empty string") {
    PagePrediction page;
    std::string md = "not-cleared";
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md.empty());

    // A page of blank recognitions is the same thing with extra steps.
    page.lines.push_back(makeLine("", 10.f, 10.f, 200.f, 30.f));
    page.lines.push_back(makeLine("   ", 10.f, 40.f, 200.f, 60.f));
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md.empty());
}

TEST_CASE("toMarkdown on a single line returns that line and nothing else") {
    PagePrediction page;
    page.lines.push_back(makeLine("Hello world", 10.f, 10.f, 200.f, 30.f));
    // No trailing newline, no heading promotion: one line IS the page median,
    // so its height ratio is exactly 1.
    CHECK(toMarkdown(page) == "Hello world");
}

TEST_CASE("lines close together and left-aligned merge into one paragraph") {
    PagePrediction page;
    page.lines.push_back(makeLine("the quick brown", 10.f, 10.f, 200.f, 30.f));
    page.lines.push_back(makeLine("fox jumps over", 10.f, 36.f, 200.f, 56.f));
    // 6px of leading against a 20px median line: 0.3 line heights, ordinary
    // single spacing. Joined with a space — the wrap point is a page-width
    // artifact, not a line break in the text.
    CHECK(toMarkdown(page) == "the quick brown fox jumps over");
}

TEST_CASE("a wide vertical gap starts a new paragraph") {
    PagePrediction page;
    page.lines.push_back(makeLine("first paragraph", 10.f, 10.f, 200.f, 30.f));
    page.lines.push_back(makeLine("second paragraph", 10.f, 90.f, 200.f, 110.f));
    // 60px of leading = 3 line heights, well past the 0.8 bar.
    CHECK(toMarkdown(page) == "first paragraph\n\nsecond paragraph");
}

TEST_CASE("a change of indent starts a new paragraph even with tight spacing") {
    PagePrediction page;
    page.lines.push_back(makeLine("flush left", 10.f, 10.f, 300.f, 26.f));
    page.lines.push_back(makeLine("indented here", 60.f, 30.f, 300.f, 46.f));
    // 4px gap (same block by the vertical test) but a 50px left shift, which is
    // 3.1 line heights — past the 1.5 indent bar.
    CHECK(toMarkdown(page) == "flush left\n\nindented here");
}

TEST_CASE("a much taller line becomes a top-level heading") {
    PagePrediction page;
    page.lines.push_back(makeLine("Invoice", 10.f, 10.f, 200.f, 60.f)); // 50px tall
    page.lines.push_back(makeLine("body one", 10.f, 100.f, 300.f, 116.f));
    page.lines.push_back(makeLine("body two", 10.f, 120.f, 300.f, 136.f));
    page.lines.push_back(makeLine("body three", 10.f, 140.f, 300.f, 156.f));
    // Median line height is 16, so the title is 3.1x body: past the 1.6 bar.
    CHECK(toMarkdown(page) == "# Invoice\n\nbody one body two body three");
}

TEST_CASE("a moderately taller line becomes a second-level heading") {
    PagePrediction page;
    page.lines.push_back(makeLine("Section", 10.f, 10.f, 200.f, 32.f)); // 22px tall
    page.lines.push_back(makeLine("body one", 10.f, 60.f, 300.f, 76.f));
    page.lines.push_back(makeLine("body two", 10.f, 80.f, 300.f, 96.f));
    page.lines.push_back(makeLine("body three", 10.f, 100.f, 300.f, 116.f));
    // 22/16 = 1.375: clears the 1.25 ascender/descender noise floor, short of
    // the 1.6 display-type bar.
    CHECK(toMarkdown(page) == "## Section\n\nbody one body two body three");
}

TEST_CASE("a whole block of large print is not promoted to a heading") {
    PagePrediction page;
    page.lines.push_back(makeLine("large print one", 10.f, 10.f, 300.f, 34.f)); // 24px
    page.lines.push_back(makeLine("large print two", 10.f, 38.f, 300.f, 62.f));
    page.lines.push_back(makeLine("large print three", 10.f, 66.f, 300.f, 90.f));
    page.lines.push_back(makeLine("body one", 10.f, 140.f, 300.f, 156.f)); // 16px
    page.lines.push_back(makeLine("body two", 10.f, 160.f, 300.f, 176.f));
    page.lines.push_back(makeLine("body three", 10.f, 180.f, 300.f, 196.f));
    page.lines.push_back(makeLine("body four", 10.f, 200.f, 300.f, 216.f));

    const std::string md = toMarkdown(page);
    // 24/16 = 1.5 would be an `##` at one or two lines, but three lines is a
    // paragraph set in large type, not a title.
    CHECK_FALSE(contains(md, "#"));
    CHECK(contains(md, "large print one large print two large print three"));
}

TEST_CASE("bullet lines stay list items instead of merging into a paragraph") {
    PagePrediction page;
    page.lines.push_back(makeLine("\xE2\x80\xA2 milk", 10.f, 10.f, 300.f, 26.f)); // U+2022
    page.lines.push_back(makeLine("* bread", 10.f, 30.f, 300.f, 46.f));
    page.lines.push_back(makeLine("\xC2\xB7 eggs", 10.f, 50.f, 300.f, 66.f)); // U+00B7
    page.lines.push_back(makeLine("- butter", 10.f, 70.f, 300.f, 86.f));
    // Every bullet shape normalizes to `-`, and the items stay adjacent lines
    // (a blank line between them would end the list).
    CHECK(toMarkdown(page) == "- milk\n- bread\n- eggs\n- butter");
}

TEST_CASE("ordered markers survive, normalized to the `1.` form") {
    PagePrediction page;
    page.lines.push_back(makeLine("1. first", 10.f, 10.f, 300.f, 26.f));
    page.lines.push_back(makeLine("2) second", 10.f, 30.f, 300.f, 46.f));
    page.lines.push_back(makeLine("a. third", 10.f, 50.f, 300.f, 66.f));
    CHECK(toMarkdown(page) == "1. first\n2. second\na. third");
}

TEST_CASE("a bullet-looking prefix without a following space is not a list") {
    PagePrediction page;
    page.lines.push_back(makeLine("-5.00 refund", 10.f, 10.f, 300.f, 26.f));
    page.lines.push_back(makeLine("1.5kg flour", 10.f, 30.f, 300.f, 46.f));
    // A negative amount and a weight, i.e. most of a receipt. Merged as prose,
    // and markdown will not read either as a list marker either.
    CHECK(toMarkdown(page) == "-5.00 refund 1.5kg flour");
}

TEST_CASE("two consecutive key/value lines become a markdown table") {
    PagePrediction page;
    LinePrediction subtotal = makeLine("Subtotal 10.00", 10.f, 10.f, 300.f, 26.f);
    subtotal.words = {
        makeWord("Subtotal", 10.f, 10.f, 80.f, 26.f),
        makeWord("10.00", 250.f, 10.f, 300.f, 26.f),
    };
    LinePrediction tax = makeLine("Tax 1.00", 10.f, 30.f, 300.f, 46.f);
    tax.words = {
        makeWord("Tax", 10.f, 30.f, 50.f, 46.f),
        makeWord("1.00", 255.f, 30.f, 300.f, 46.f),
    };
    page.lines.push_back(subtotal);
    page.lines.push_back(tax);

    // 170px of gutter against a 16px median line is 10.6 line heights.
    CHECK(toMarkdown(page) ==
          "| Subtotal | 10.00 |\n| --- | --- |\n| Tax | 1.00 |");
}

TEST_CASE("the same two lines with no word boxes stay plain paragraph text") {
    PagePrediction page;
    // Byte-for-byte the fixture above with `words` left empty, which is the
    // default: EngineConfig::returnWordBoxes is off unless a caller asks.
    page.lines.push_back(makeLine("Subtotal 10.00", 10.f, 10.f, 300.f, 26.f));
    page.lines.push_back(makeLine("Tax 1.00", 10.f, 30.f, 300.f, 46.f));

    std::string md;
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md == "Subtotal 10.00 Tax 1.00");
    CHECK_FALSE(contains(md, "|"));
    CHECK_FALSE(contains(md, "---"));
}

TEST_CASE("an isolated key/value line is a bold label, not a one-row table") {
    PagePrediction page;
    LinePrediction row = makeLine("Item name 12.00", 10.f, 10.f, 300.f, 26.f);
    row.words = {
        makeWord("Item", 10.f, 10.f, 50.f, 26.f),
        makeWord("name", 55.f, 10.f, 100.f, 26.f),
        makeWord("12.00", 250.f, 10.f, 300.f, 26.f),
    };
    page.lines.push_back(row);

    // The 150px gutter is 30x the 5px word space next to it, so the split is
    // real; a one-row GFM table would be a header plus a separator and no body.
    CHECK(toMarkdown(page) == "**Item name** 12.00");
}

TEST_CASE("evenly spaced words are not split, however wide the spacing") {
    PagePrediction page;
    LinePrediction line = makeLine("one two three four", 10.f, 10.f, 400.f, 26.f);
    line.words = {
        makeWord("one", 10.f, 10.f, 60.f, 26.f),
        makeWord("two", 100.f, 10.f, 150.f, 26.f),
        makeWord("three", 190.f, 10.f, 250.f, 26.f),
        makeWord("four", 320.f, 10.f, 380.f, 26.f),
    };
    page.lines.push_back(line);

    // Gaps 40/40/70: the widest clears the absolute bar but is under 3x the
    // others, which is what keeps justified prose out of the table path.
    std::string md;
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md == "one two three four");
}

// --- line-level rows --------------------------------------------------------
// A gutter wide enough to matter is wide enough that DBNet emits two separate
// boxes, so the label and the value arrive as two consecutive LinePredictions
// on one visual row and there are no word boxes involved anywhere below.

namespace {

// Invoice fields at a ~360px gutter, rows 24px apart (1.5 line heights of
// leading, more air than prose but one table).
PagePrediction fieldsPage(float k) {
    PagePrediction page;
    page.lines.push_back(makeLine("Invoice No", 40.f * k, 100.f * k, 160.f * k, 116.f * k));
    page.lines.push_back(makeLine("INV-2026-0042", 520.f * k, 100.f * k, 700.f * k, 116.f * k));
    page.lines.push_back(makeLine("Date", 40.f * k, 140.f * k, 90.f * k, 156.f * k));
    page.lines.push_back(makeLine("07 August 2026", 520.f * k, 140.f * k, 680.f * k, 156.f * k));
    return page;
}

} // namespace

TEST_CASE("two same-row line pairs become a table") {
    // Without line-level pairing each half became its own paragraph, because
    // the left-edge rule filed x=40 and x=520 as different blocks.
    // The 24px inter-row gap is past the 12.8px prose bar and under the 32px
    // form-row bar, so this also pins kRowGapFraction.
    CHECK(toMarkdown(fieldsPage(1.0f)) ==
          "| Invoice No | INV-2026-0042 |\n"
          "| --- | --- |\n"
          "| Date | 07 August 2026 |");
}

TEST_CASE("one isolated same-row pair is a bold label, not a one-row table") {
    PagePrediction page;
    page.lines.push_back(makeLine("Invoice No", 40.f, 100.f, 160.f, 116.f));
    page.lines.push_back(makeLine("INV-2026-0042", 520.f, 100.f, 700.f, 116.f));
    CHECK(toMarkdown(page) == "**Invoice No** INV-2026-0042");
}

TEST_CASE("same-row pairs at 1x and 4x produce byte-identical markdown") {
    // The new path measures a vertical overlap and a horizontal gutter, both of
    // which are pixel counts until they are divided by the median line height.
    const std::string expected =
        "| Invoice No | INV-2026-0042 |\n"
        "| --- | --- |\n"
        "| Date | 07 August 2026 |";
    CHECK(toMarkdown(fieldsPage(1.0f)) == expected);
    CHECK(toMarkdown(fieldsPage(4.0f)) == expected);
    CHECK(toMarkdown(fieldsPage(1.0f)) == toMarkdown(fieldsPage(4.0f)));
}

TEST_CASE("a same-row pair only slightly apart is not a key/value row") {
    PagePrediction page;
    page.lines.push_back(makeLine("Invoice No", 40.f, 100.f, 160.f, 116.f));
    page.lines.push_back(makeLine("INV-2026-0042", 170.f, 100.f, 300.f, 116.f));

    // 10px between the boxes is 0.6 line heights — a word space, i.e. the shape
    // of a phrase the detector over-split, not a gutter. No pairing; the
    // left-edge rule then keeps them apart as before.
    std::string md;
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md == "Invoice No\n\nINV-2026-0042");
    CHECK_FALSE(contains(md, "|"));
    CHECK_FALSE(contains(md, "**"));
}

TEST_CASE("three boxes on a row split at the widest gutter, into two columns") {
    PagePrediction page;
    page.lines.push_back(makeLine("1", 40.f, 100.f, 60.f, 116.f));
    page.lines.push_back(makeLine("Cheeseburger", 70.f, 100.f, 220.f, 116.f));
    page.lines.push_back(makeLine("12.00", 520.f, 100.f, 580.f, 116.f));
    page.lines.push_back(makeLine("2", 40.f, 140.f, 60.f, 156.f));
    page.lines.push_back(makeLine("Fries", 70.f, 140.f, 150.f, 156.f));
    page.lines.push_back(makeLine("4.50", 520.f, 140.f, 580.f, 156.f));

    // Gutters 10 and 300: quantity hugs the description, the amount is far
    // right, so the split lands after the description. Capped at two columns by
    // design — a third cell would need a column count agreed across the run.
    CHECK(toMarkdown(page) ==
          "| 1 Cheeseburger | 12.00 |\n"
          "| --- | --- |\n"
          "| 2 Fries | 4.50 |");
}

TEST_CASE("stacked lines never pair, however far apart their left edges are") {
    PagePrediction page;
    page.lines.push_back(makeLine("left column line", 40.f, 100.f, 300.f, 116.f));
    page.lines.push_back(makeLine("indented reply", 520.f, 130.f, 700.f, 146.f));

    // Vertical extents do not overlap at all, so the row test rejects them
    // before the gutter is ever measured.
    std::string md;
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md == "left column line\n\nindented reply");
}

TEST_CASE("the same page at 1x and 4x produces byte-identical markdown") {
    // The property that catches any absolute-pixel threshold: a 300 DPI scan
    // and a 96 DPI screenshot of one page must reconstruct the same document.
    auto build = [](float k) {
        PagePrediction page;
        page.lines.push_back(makeLine("Receipt", 10.f * k, 10.f * k, 200.f * k, 60.f * k));
        page.lines.push_back(makeLine("thank you for", 10.f * k, 100.f * k, 300.f * k, 116.f * k));
        page.lines.push_back(makeLine("shopping with us", 10.f * k, 120.f * k, 300.f * k, 136.f * k));
        page.lines.push_back(makeLine("- milk", 10.f * k, 160.f * k, 300.f * k, 176.f * k));
        page.lines.push_back(makeLine("- bread", 10.f * k, 180.f * k, 300.f * k, 196.f * k));

        LinePrediction total = makeLine("TOTAL 12.00", 10.f * k, 220.f * k, 300.f * k, 236.f * k);
        total.words = {
            makeWord("TOTAL", 10.f * k, 220.f * k, 60.f * k, 236.f * k),
            makeWord("12.00", 250.f * k, 220.f * k, 300.f * k, 236.f * k),
        };
        LinePrediction tax = makeLine("TAX 1.00", 10.f * k, 240.f * k, 300.f * k, 256.f * k);
        tax.words = {
            makeWord("TAX", 10.f * k, 240.f * k, 50.f * k, 256.f * k),
            makeWord("1.00", 260.f * k, 240.f * k, 300.f * k, 256.f * k),
        };
        page.lines.push_back(total);
        page.lines.push_back(tax);
        return page;
    };

    const std::string expected =
        "# Receipt\n"
        "\n"
        "thank you for shopping with us\n"
        "\n"
        "- milk\n"
        "- bread\n"
        "\n"
        "| TOTAL | 12.00 |\n"
        "| --- | --- |\n"
        "| TAX | 1.00 |";

    // Pinned at 1x so a scale-invariant *wrong* answer cannot pass this test.
    CHECK(toMarkdown(build(1.0f)) == expected);
    CHECK(toMarkdown(build(4.0f)) == expected);
    CHECK(toMarkdown(build(1.0f)) == toMarkdown(build(4.0f)));
}

TEST_CASE("pipes and asterisks in body text are left alone") {
    PagePrediction page;
    page.lines.push_back(makeLine("A | B * C **** 1234", 10.f, 10.f, 300.f, 26.f));
    // Outside a table `|` is an ordinary character, and an unpaired `*` run is
    // literal — escaping either would only add backslashes to a card number.
    CHECK(toMarkdown(page) == "A | B * C **** 1234");
}

TEST_CASE("a pipe inside a table cell is escaped") {
    PagePrediction page;
    LinePrediction first = makeLine("Ref 10|00", 10.f, 10.f, 300.f, 26.f);
    first.words = {
        makeWord("Ref", 10.f, 10.f, 50.f, 26.f),
        makeWord("10|00", 250.f, 10.f, 300.f, 26.f),
    };
    LinePrediction second = makeLine("Tax 1.00", 10.f, 30.f, 300.f, 46.f);
    second.words = {
        makeWord("Tax", 10.f, 30.f, 50.f, 46.f),
        makeWord("1.00", 255.f, 30.f, 300.f, 46.f),
    };
    page.lines.push_back(first);
    page.lines.push_back(second);

    const std::string md = toMarkdown(page);
    CHECK(contains(md, "| Ref | 10\\|00 |"));
    // The cell did not end early: the row still has exactly the two columns the
    // separator row promises.
    CHECK(contains(md, "\n| --- | --- |\n"));
}

TEST_CASE("a leading hash does not silently become a heading") {
    PagePrediction page;
    page.lines.push_back(makeLine("#1 item", 10.f, 10.f, 300.f, 26.f));
    CHECK(toMarkdown(page) == "\\#1 item");
}

TEST_CASE("a dashed rule cannot retitle the paragraph printed above it") {
    PagePrediction page;
    page.lines.push_back(makeLine("Items", 10.f, 10.f, 200.f, 26.f));
    page.lines.push_back(makeLine("--------", 10.f, 100.f, 200.f, 116.f));
    // Unescaped, `--------` is a setext underline and turns "Items" into an H2.
    CHECK(toMarkdown(page) == "Items\n\n\\--------");
}

TEST_CASE("a link-shaped run of OCR text keeps its brackets") {
    PagePrediction page;
    page.lines.push_back(makeLine("see [1] and [2](3) below", 10.f, 10.f, 400.f, 26.f));
    // `[1]` is a shortcut reference with no definition, so it renders literally
    // and is left as-is; only the `](` seam is broken.
    CHECK(toMarkdown(page) == "see [1] and [2\\](3) below");
}

TEST_CASE("degenerate polygons and empty text do not crash") {
    PagePrediction page;
    page.lines.push_back(LinePrediction{}); // no polygon, no text

    LinePrediction noPolygon;
    noPolygon.text = "no polygon";
    page.lines.push_back(noPolygon);

    page.lines.push_back(makeLine("flat", 10.f, 10.f, 200.f, 10.f)); // zero height
    page.lines.push_back(makeLine("   ", 10.f, 30.f, 200.f, 50.f));  // whitespace only

    LinePrediction onePoint;
    onePoint.polygon = {{5.f, 5.f}};
    onePoint.text = "point";
    page.lines.push_back(onePoint);

    LinePrediction emptyWordBoxes = makeLine("k v", 10.f, 60.f, 200.f, 80.f);
    emptyWordBoxes.words = {WordBox{}, WordBox{}}; // words with no geometry
    page.lines.push_back(emptyWordBoxes);

    std::string md;
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(contains(md, "no polygon"));
    CHECK(contains(md, "flat"));
    CHECK(contains(md, "point"));
    CHECK(contains(md, "k v"));
    CHECK_FALSE(contains(md, "|")); // no table conjured out of geometry-less words
}

TEST_CASE("a page whose polygons are all missing still returns its text") {
    PagePrediction page;
    LinePrediction a;
    a.text = "alpha";
    LinePrediction b;
    b.text = "beta";
    page.lines.push_back(a);
    page.lines.push_back(b);

    // Median line height is 0 here, so every threshold is 0 and nothing may
    // divide by it. Lines with no geometry stay with whatever they followed.
    std::string md;
    CHECK_NOTHROW(md = toMarkdown(page));
    CHECK(md == "alpha beta");
}
