// Layout reconstruction: recognized lines -> markdown blocks. Geometry only,
// no model and no page segmentation — see markdown.hpp for the scope statement.
//
// Key/value rows arrive in two shapes and there is a detector for each. A
// narrow gutter leaves the label and the value inside ONE detector box, so the
// gap shows up between that line's `words`; a wide gutter makes DBNet emit two
// separate boxes, so the gap shows up between two consecutive LinePredictions
// that share a visual row. Both feed splitAtWidestGap and both come out as the
// same Kind::KeyValue item, so there is one set of thresholds and one emitter.
//
// Every threshold below is a fraction of the page's MEDIAN LINE HEIGHT, never a
// pixel count. sortLinesReadingOrder learned that with a hard-coded 12px row
// tolerance which fragmented 300 DPI scans and merged thumbnails; the same page
// photographed at 96 DPI and scanned at 300 DPI has to produce the same
// markdown, and the median line height is the only unit that delivers that.
#include "arboOCR/markdown.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace arbo::ocr {

namespace {

// Vertical whitespace between two lines of one paragraph is the line pitch
// minus the glyph height: roughly 0.2-0.5 line heights for single-spaced text
// (and occasionally negative, since detector boxes are unclipped and can
// overlap). A real paragraph break adds a whole blank line, so it starts at
// ~1.2. 0.8 sits in the empty middle of that range.
constexpr float kBlockGapFraction = 0.8f;

// ...but form and invoice rows are set with far more air than prose — two to
// three times line pitch is ordinary for a field list — so between two
// key/value rows the bar has to be higher, or every field would be filed as its
// own block and come out as a lone **label** instead of joining the table. A
// genuine break between two groups of fields is wider still.
constexpr float kRowGapFraction = 2.0f;

// Left edges of lines inside one paragraph agree to within detector jitter,
// well under half a line height. A deliberate indent is an em quad or a tab
// stop — one to several line heights. 1.5 separates the two, and is loose
// enough that the per-line wobble of a centred receipt header does not shred
// the header into one block per line.
constexpr float kIndentFraction = 1.5f;

// Two boxes sit on the same visual row when their vertical extents overlap by
// at least half a line. Stacked paragraph lines overlap by a small fraction of
// that at most (unclipped detector boxes touch, sometimes graze), while a real
// label/value pair overlaps by nearly a full line height — there is a wide
// empty band between the two populations and 0.5 sits in it. Overlap rather
// than equal centroids because real boxes jitter, and because a value set in a
// slightly larger face still overlaps its label completely.
constexpr float kRowOverlapFraction = 0.5f;

// Heading tiers, as the ratio of a block's line height to the page median.
// A line box's height moves by ~15% on ascenders and descenders alone (an
// all-caps line, or one with no descender, measures short at the same point
// size), so the lower tier has to clear that noise: 1.25. Display type in a
// document is conventionally 1.5-2x body, hence 1.6 for the top tier.
constexpr float kHeading1Ratio = 1.6f;
constexpr float kHeading2Ratio = 1.25f;

// A heading is a title, not a page of large print. Past two lines a tall block
// is a large-print paragraph and promoting it would swallow the body text.
constexpr size_t kMaxHeadingLines = 2;

// A key/value gap has to be whitespace no word spacing could explain. A space
// glyph is a quarter to a third of an em, i.e. ~0.3 line heights; 1.5 line
// heights of blank is four to six spaces wide — a column gutter, not a space.
// One bar serves both detectors: a fragment the detector over-split out of one
// phrase ("Invoice" / "No") sits a word space apart and is rejected either way,
// while a real gutter clears it either way.
constexpr float kKeyValueGapFraction = 1.5f;

// ...and, when the row has enough pieces to have a "typical" gap at all, the
// candidate must also dwarf it. This is what keeps fully justified prose (where
// every inter-word space is stretched, so all gaps grow together) out of the
// table path, and what makes a three-piece line item split after the
// description rather than after the quantity.
constexpr float kKeyValueGapRatio = 3.0f;

// U+2022 BULLET and U+00B7 MIDDLE DOT are what the recognizers actually emit
// for a printed dot; the two ASCII ones are what a plain-text export looks like.
const std::string kBullets[] = {"-", "*", "\xE2\x80\xA2", "\xC2\xB7"};

struct Box {
    float x0 = 0.f;
    float y0 = 0.f;
    float x1 = 0.f;
    float y1 = 0.f;
    bool valid = false;
    float height() const { return y1 - y0; }
};

Box boundingBox(const Polygon& poly) {
    Box box;
    if (poly.empty()) return box; // guard: no geometry at all, stays invalid
    box.x0 = box.x1 = poly.front().x;
    box.y0 = box.y1 = poly.front().y;
    for (const auto& pt : poly) {
        box.x0 = std::min(box.x0, pt.x);
        box.x1 = std::max(box.x1, pt.x);
        box.y0 = std::min(box.y0, pt.y);
        box.y1 = std::max(box.y1, pt.y);
    }
    box.valid = true;
    return box;
}

// Same nth_element idiom sortLinesReadingOrder uses for its row tolerance: a
// median shrugs off the one absurd box a detector occasionally emits, where a
// mean would drag every threshold on the page with it. Takes its vector by
// value because nth_element reorders what it is given.
float medianOf(std::vector<float> values) {
    if (values.empty()) return 0.f;
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

bool isSpace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

std::string trim(const std::string& s) {
    size_t begin = 0;
    size_t end = s.size();
    while (begin < end && isSpace(s[begin])) ++begin;
    while (end > begin && isSpace(s[end - 1])) --end;
    return s.substr(begin, end - begin);
}

/// Escaping policy: touch only what would silently change the document, and
/// leave everything else as the recognizer read it. Concretely:
///   * `|` is escaped inside table cells, where it ends the cell, and nowhere
///     else, where it is an ordinary character.
///   * `]` is escaped only when `](` or `][` follows — the one shape that turns
///     OCR text into a link and eats the target. A bare `[1]` is a shortcut
///     reference with no definition, which renders literally, so citations and
///     footnote markers stay readable instead of becoming `\[1\]`.
///   * CR/LF/TAB become a space. They are the only inline characters that can
///     shatter block structure from inside a line.
///   * `*`, `_`, `#` and backticks are NOT escaped mid-line. Unpaired they are
///     literal; paired they cost at worst italics or a code span, with the text
///     still on the page — while escaping them turns the masked card number
///     "**** 1234" that is on half the receipts we target into "\*\*\*\* 1234".
///     Structure-forming positions (start of a line) are handled separately by
///     escapeBlockStart, which is where the real damage would happen.
std::string escapeInline(const std::string& text, bool inTableCell) {
    std::string out;
    out.reserve(text.size() + 4);
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '\n' || c == '\r' || c == '\t') {
            out += ' ';
        } else if (c == '|' && inTableCell) {
            out += "\\|";
        } else if (c == ']' && i + 1 < text.size() && (text[i + 1] == '(' || text[i + 1] == '[')) {
            out += "\\]";
        } else {
            out += c;
        }
    }
    return out;
}

/// Backslash the first character when it would open a block we did not ask for:
/// an ATX heading, a blockquote, a bullet, or a rule line. The rule case is the
/// one that actually bites — a receipt's `--------` separator is a setext
/// underline, so left alone it silently retitles the paragraph printed above it.
std::string escapeBlockStart(const std::string& text) {
    size_t i = 0;
    while (i < text.size() && text[i] == ' ') ++i;
    if (i >= text.size()) return text;

    const char c = text[i];
    bool needsEscape = (c == '#' || c == '>');
    if (!needsEscape && (c == '-' || c == '*' || c == '+') &&
        (i + 1 == text.size() || text[i + 1] == ' ')) {
        // `+ x` is a bullet listMarker deliberately does not claim; `-`/`*` are
        // already claimed, so for them this is belt and braces.
        needsEscape = true;
    }
    if (!needsEscape && (c == '-' || c == '=' || c == '_' || c == '*')) {
        bool onlyRule = true;
        for (size_t j = i; j < text.size() && onlyRule; ++j) {
            onlyRule = (text[j] == c || text[j] == ' ');
        }
        needsEscape = onlyRule;
    }
    if (!needsEscape) return text;
    return text.substr(0, i) + "\\" + text.substr(i);
}

/// Normalized list marker for `text`, or an empty string when it is not a list
/// item; `rest` receives the text after the marker. Markers must be followed by
/// a space or end the line — `-5.00` on a receipt is a negative amount, not a
/// bullet, and `1.5kg` is not item one.
std::string listMarker(const std::string& text, std::string& rest) {
    size_t i = 0;
    while (i < text.size() && text[i] == ' ') ++i;
    if (i >= text.size()) return {};

    for (const std::string& bullet : kBullets) {
        if (text.compare(i, bullet.size(), bullet) == 0 &&
            (i + bullet.size() == text.size() || text[i + bullet.size()] == ' ')) {
            rest = trim(text.substr(i + bullet.size()));
            return "-"; // every bullet shape collapses to the one markdown spells
        }
    }

    // Ordered: a digit run or a single letter, then '.' or ')'. No cap on the
    // digit count, deliberately — markdown numbers a list from its first marker
    // and we re-emit the marker verbatim, so even a misread "2024. 01. 01" date
    // comes back out as the same characters that went in.
    size_t j = i;
    while (j < text.size() && text[j] >= '0' && text[j] <= '9') ++j;
    if (j == i && ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))) {
        j = i + 1;
    }
    if (j > i && j < text.size() && (text[j] == '.' || text[j] == ')') &&
        (j + 1 == text.size() || text[j + 1] == ' ')) {
        rest = trim(text.substr(j + 1));
        return text.substr(i, j - i) + "."; // `2)` -> `2.`, the form markdown counts
    }
    return {};
}

// ponytail: a single space between pieces is right for space-delimited scripts
// and wrong for CJK, where groupTokensIntoWords gives every character its own
// WordBox. Joining only when both sides of the seam are ASCII gets both cases
// right without carrying a Unicode word-break table.
std::string joinPieces(const std::vector<std::string>& texts, size_t begin, size_t end) {
    std::string out;
    for (size_t i = begin; i < end && i < texts.size(); ++i) {
        if (texts[i].empty()) continue;
        if (!out.empty() && static_cast<unsigned char>(out.back()) < 0x80 &&
            static_cast<unsigned char>(texts[i].front()) < 0x80) {
            out += ' ';
        }
        out += texts[i];
    }
    return out;
}

/// Split one visual row into a label and a value at its single conspicuously
/// wide internal gap — the shape of every receipt total and every invoice
/// field. Shared by both detectors: the word-level one passes the WordBoxes
/// inside one LinePrediction, the line-level one passes the LinePredictions
/// that share a row. Both ask the same question (is this whitespace a gutter or
/// a space?), so both get the same answer from the same two thresholds.
///
/// ponytail: two columns out, always. Everything left of the widest gap becomes
/// the label and everything right of it the value, so a three-piece line item
/// comes back as "1 Cheeseburger" / "12.00" rather than as three cells. Agreeing
/// a column count across the rows of a run is cell-grid reconstruction, which
/// markdown.hpp puts out of scope; capping at two never strands a piece.
bool splitAtWidestGap(const std::vector<Box>& boxes, const std::vector<std::string>& texts,
                      float medianHeight, std::string& label, std::string& value) {
    if (medianHeight <= 0.f) {
        return false; // guard: no page scale to measure a "wide" gap against
    }
    if (boxes.size() < 2 || boxes.size() != texts.size()) return false;

    std::vector<float> gaps;
    gaps.reserve(boxes.size() - 1);
    for (size_t i = 1; i < boxes.size(); ++i) {
        // Word boxes come from CTC alignment and lag right by about half a
        // character (types.hpp), so neighbours can overlap by a hair. Clamp.
        gaps.push_back(std::max(0.f, boxes[i].x0 - boxes[i - 1].x1));
    }

    size_t at = 0;
    for (size_t i = 1; i < gaps.size(); ++i) {
        if (gaps[i] > gaps[at]) at = i;
    }
    const float widest = gaps[at];
    if (widest <= kKeyValueGapFraction * medianHeight) return false;

    // Compare against the median of the OTHER gaps, not of all of them: with
    // exactly two gaps the upper median IS the widest, so an all-gaps median
    // would make this test unfireable on any three-piece row.
    std::vector<float> others;
    others.reserve(gaps.size());
    for (size_t i = 0; i < gaps.size(); ++i) {
        if (i != at) others.push_back(gaps[i]);
    }
    if (!others.empty() && widest <= kKeyValueGapRatio * medianOf(others)) return false;

    label = joinPieces(texts, 0, at + 1);
    value = joinPieces(texts, at + 1, texts.size());
    return !label.empty() && !value.empty();
}

/// Word-level detector: the label and the value ended up inside ONE detector
/// box, which is what a narrow gutter produces. Empty `words` — the default,
/// since EngineConfig::returnWordBoxes is off unless a caller asks — simply
/// means no split here, which is why word boxes are an enhancement and not a
/// precondition: the line-level detector below needs none at all.
bool splitLineWords(const LinePrediction& line, float medianHeight,
                    std::string& label, std::string& value) {
    std::vector<Box> boxes;
    std::vector<std::string> texts;
    boxes.reserve(line.words.size());
    texts.reserve(line.words.size());
    for (const auto& word : line.words) {
        Box box = boundingBox(word.polygon);
        if (!box.valid || word.text.empty()) continue;
        boxes.push_back(box);
        texts.push_back(word.text);
    }
    return splitAtWidestGap(boxes, texts, medianHeight, label, value);
}

enum class Kind { Text, List, KeyValue };

struct Item {
    Kind kind = Kind::Text;
    std::string text;   // whole line (Text), or the part after the marker (List)
    std::string marker; // normalized list marker (List)
    std::string label;  // left of the wide gap (KeyValue)
    std::string value;  // right of the wide gap (KeyValue)
    Box box;
};

/// Do these two lines sit on the same visual row, with the second starting
/// after the first ends? Vertical overlap is the test rather than equal
/// centroids: real boxes jitter, and sortLinesReadingOrder already grouped the
/// page into rows within a centroid tolerance, so same-row lines are guaranteed
/// adjacent here and all this has to do is confirm what the sort assumed.
bool sharesRow(const Item& a, const Item& b, float medianHeight) {
    // A list item, or a line the word-level pass already split, carries its own
    // structure; gaps around it are not this function's business.
    if (a.kind != Kind::Text || b.kind != Kind::Text) return false;
    if (!a.box.valid || !b.box.valid) return false;
    const float overlap = std::min(a.box.y1, b.box.y1) - std::max(a.box.y0, b.box.y0);
    if (overlap <= kRowOverlapFraction * medianHeight) return false;
    return b.box.x0 >= a.box.x1; // no horizontal overlap: b starts after a ends
}

/// Line-level detector: a real gutter is wide enough that DBNet emits the label
/// and the value as two SEPARATE boxes, so the gap never appears between one
/// line's words and the word-level pass cannot see it at all. Fold each run of
/// boxes that share a visual row into a single key/value Item, which then flows
/// through the same block grouping, the same table run and the same emitter as
/// the word-level case.
///
/// This has to run before block grouping: the left-edge rule there would
/// otherwise (correctly, by its own lights) file a label at x=40 and its value
/// at x=520 as two different blocks, which is exactly how a wide-gutter invoice
/// came out as alternating one-line paragraphs.
///
/// ponytail: this fires on a two-column PAGE body too, and a whole page of them
/// would come out as one giant table. No per-row geometry can separate the two
/// cases — a ragged-right left column leaves gaps every bit as wide as an
/// invoice gutter, so neither the absolute gutter width nor its share of the row
/// discriminates. A page-level "nearly every line pairs up" guard would catch
/// articles, but it would equally break a full-page form, which IS the shape
/// arboOCR targets, so the miss stays on the article side: multi-column pages
/// are already out of scope in markdown.hpp, and sortLinesReadingOrder
/// interleaves their columns before this file ever sees them.
std::vector<Item> pairRows(const std::vector<Item>& items, float medianHeight) {
    std::vector<Item> out;
    out.reserve(items.size());
    for (size_t i = 0; i < items.size();) {
        size_t j = i;
        while (j + 1 < items.size() && sharesRow(items[j], items[j + 1], medianHeight)) ++j;
        if (j > i) {
            std::vector<Box> boxes;
            std::vector<std::string> texts;
            boxes.reserve(j - i + 1);
            texts.reserve(j - i + 1);
            for (size_t k = i; k <= j; ++k) {
                boxes.push_back(items[k].box);
                texts.push_back(items[k].text);
            }
            Item row;
            if (splitAtWidestGap(boxes, texts, medianHeight, row.label, row.value)) {
                row.kind = Kind::KeyValue;
                row.box = boxes.front();
                for (const auto& box : boxes) {
                    row.box.x0 = std::min(row.box.x0, box.x0);
                    row.box.y0 = std::min(row.box.y0, box.y0);
                    row.box.x1 = std::max(row.box.x1, box.x1);
                    row.box.y1 = std::max(row.box.y1, box.y1);
                }
                out.push_back(std::move(row));
                i = j + 1;
                continue;
            }
        }
        // No usable gutter across this row: keep the line as it was and retry
        // from the next one, so a failed three-piece row can still pair its tail.
        out.push_back(items[i]);
        ++i;
    }
    return out;
}

/// Heading tier for a block, or 0 when it is body text.
int headingLevel(const std::vector<Item>& items, float medianHeight) {
    if (medianHeight <= 0.f || items.empty() || items.size() > kMaxHeadingLines) return 0;
    std::vector<float> heights;
    heights.reserve(items.size());
    for (const auto& item : items) {
        // A list item or a key/value row already has structure of its own, and
        // large print does not outrank it.
        if (item.kind != Kind::Text || !item.box.valid) return 0;
        heights.push_back(item.box.height());
    }
    const float height = medianOf(heights);
    if (height >= kHeading1Ratio * medianHeight) return 1;
    if (height >= kHeading2Ratio * medianHeight) return 2;
    return 0;
}

std::string tableRow(const std::string& label, const std::string& value) {
    return "| " + escapeInline(label, true) + " | " + escapeInline(value, true) + " |";
}

// One run of same-kind lines inside a block becomes one chunk. Chunks are
// joined with a blank line by the caller, which is what keeps a list glued
// together (its items are one chunk) but separated from the prose above it.
void appendSegment(std::vector<std::string>& chunks, const std::vector<Item>& items,
                   size_t begin, size_t end) {
    if (begin >= end) return;

    if (items[begin].kind == Kind::List) {
        std::string chunk;
        for (size_t i = begin; i < end; ++i) {
            if (!chunk.empty()) chunk += '\n';
            chunk += items[i].marker;
            const std::string body = escapeBlockStart(escapeInline(items[i].text, false));
            if (!body.empty()) chunk += " " + body;
        }
        chunks.push_back(chunk);
        return;
    }

    if (items[begin].kind == Kind::KeyValue) {
        if (end - begin == 1) {
            // A lone wide gap is weak evidence, and a one-row GFM table is a
            // header plus a separator with no body — it renders as an empty
            // table, strictly worse than the text. Bolding the label keeps both
            // halves and the split without claiming a column structure exists.
            chunks.push_back("**" + escapeInline(items[begin].label, false) + "** " +
                             escapeInline(items[begin].value, false));
            return;
        }
        // GFM has no table without a header row, so the first pair becomes one.
        // Synthesizing an empty header would render as a blank band above the
        // data; promoting the first row invents no text that was not on the
        // page, and on invoices that row genuinely is the column caption more
        // often than not.
        std::string chunk = tableRow(items[begin].label, items[begin].value);
        chunk += "\n| --- | --- |";
        for (size_t i = begin + 1; i < end; ++i) {
            chunk += "\n" + tableRow(items[i].label, items[i].value);
        }
        chunks.push_back(chunk);
        return;
    }

    // Plain text: one space, not a hard break. Where the recognizer wrapped is
    // an artifact of the page width, not of the sentence.
    std::string joined;
    for (size_t i = begin; i < end; ++i) {
        if (!joined.empty()) joined += ' ';
        joined += escapeInline(items[i].text, false);
    }
    chunks.push_back(escapeBlockStart(joined));
}

void appendBlock(std::vector<std::string>& chunks, const std::vector<Item>& items,
                 float medianHeight) {
    const int level = headingLevel(items, medianHeight);
    if (level > 0) {
        std::string text;
        for (const auto& item : items) {
            if (!text.empty()) text += ' ';
            text += escapeInline(item.text, false);
        }
        chunks.push_back(std::string(static_cast<size_t>(level), '#') + " " + text);
        return;
    }
    for (size_t i = 0; i < items.size();) {
        size_t j = i;
        while (j < items.size() && items[j].kind == items[i].kind) ++j;
        appendSegment(chunks, items, i, j);
        i = j;
    }
}

} // namespace

std::string toMarkdown(const PagePrediction& page) {
    // First pass: keep the lines that have something to render, and measure the
    // page. Empty lines are dropped here rather than later so that they cannot
    // invent a vertical gap between the two real lines they sit between.
    std::vector<Item> items;
    std::vector<size_t> sources;
    std::vector<float> heights;
    items.reserve(page.lines.size());
    sources.reserve(page.lines.size());
    heights.reserve(page.lines.size());
    for (size_t i = 0; i < page.lines.size(); ++i) {
        Item item;
        item.text = trim(page.lines[i].text);
        if (item.text.empty()) continue;
        item.box = boundingBox(page.lines[i].polygon);
        if (item.box.valid) heights.push_back(item.box.height());
        items.push_back(std::move(item));
        sources.push_back(i);
    }
    if (items.empty()) return {};

    // The one unit every threshold in this file is expressed in. Measured from
    // single lines, before row pairing widens any box. Zero when no line had a
    // usable polygon, which the guards downstream all check for.
    const float medianHeight = medianOf(heights);

    // Second pass: classify. Needs medianHeight, hence its own loop.
    for (size_t i = 0; i < items.size(); ++i) {
        std::string rest;
        std::string marker = listMarker(items[i].text, rest);
        if (!marker.empty()) {
            items[i].kind = Kind::List;
            items[i].marker = marker;
            items[i].text = rest;
            continue;
        }
        if (splitLineWords(page.lines[sources[i]], medianHeight, items[i].label, items[i].value)) {
            items[i].kind = Kind::KeyValue;
        }
    }

    // Third pass: the same split one level up — across lines that share a row
    // rather than across words that share a line.
    items = pairRows(items, medianHeight);

    // Blocks: a wide vertical gap ends one, and so does a change of indent.
    std::vector<std::vector<Item>> blocks;
    for (size_t i = 0; i < items.size(); ++i) {
        bool startNew = (i == 0);
        if (i > 0 && items[i - 1].box.valid && items[i].box.valid) {
            const bool bothRows =
                items[i - 1].kind == Kind::KeyValue && items[i].kind == Kind::KeyValue;
            const float gap = items[i].box.y0 - items[i - 1].box.y1;
            const float shift = std::fabs(items[i].box.x0 - items[i - 1].box.x0);
            startNew = gap > (bothRows ? kRowGapFraction : kBlockGapFraction) * medianHeight ||
                       shift > kIndentFraction * medianHeight;
        }
        // ponytail: a line with no polygon has no geometry to compare, so it
        // stays with whatever it followed rather than inventing a break by
        // pretending it sits at the origin.
        if (startNew) blocks.emplace_back();
        blocks.back().push_back(items[i]);
    }

    std::vector<std::string> chunks;
    for (const auto& block : blocks) {
        appendBlock(chunks, block, medianHeight);
    }

    std::string out;
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (i) out += "\n\n";
        out += chunks[i];
    }
    return out;
}

} // namespace arbo::ocr
