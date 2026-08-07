#pragma once
// Layout reconstruction: recognized lines -> a rough markdown document.

#include <string>

#include "arboOCR/types.hpp"

namespace arbo::ocr {

/// Reconstruct a markdown document from a recognized page.
///
/// OCR gives you lines, not structure. This infers the structure back from
/// line geometry: lines close together and left-aligned become one paragraph,
/// a taller-than-usual line becomes a heading, bullet and numeric prefixes
/// stay list items, and a line split by a wide internal gap becomes a
/// `key | value` table row (receipts and invoices are mostly this).
///
/// Rough by design, and rough in the same places RapidOCR's to_markdown is:
/// no multi-column page splitting, no cell-grid table reconstruction, no
/// image or rule detection. Heuristics are geometric, so they degrade on
/// heavily skewed or curved text rather than failing loudly.
///
/// `page.lines` is expected in reading order, which Engine::recognize
/// already guarantees. Never throws; an empty page yields an empty string.
std::string toMarkdown(const PagePrediction& page);

} // namespace arbo::ocr
