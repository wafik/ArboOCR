#include "arboOCR/types.hpp"

#include <iomanip>
#include <sstream>

namespace arbo::ocr {

namespace {

std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                std::ostringstream hex;
                hex << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c);
                out += hex.str();
            } else {
                out += static_cast<char>(c);
            }
            break;
        }
    }
    return out;
}

void appendPolygon(std::ostringstream& os, const Polygon& poly, bool pretty, int indent) {
    const std::string pad(static_cast<size_t>(indent), ' ');
    const std::string padIn(static_cast<size_t>(indent + 2), ' ');
    const std::string padPt(static_cast<size_t>(indent + 4), ' ');
    if (pretty) {
        os << "[\n";
        for (size_t i = 0; i < poly.size(); i++) {
            os << padPt << "{\"x\":" << poly[i].x << ",\"y\":" << poly[i].y << "}";
            if (i + 1 < poly.size()) os << ",";
            os << "\n";
        }
        os << padIn << "]";
    } else {
        os << "[";
        for (size_t i = 0; i < poly.size(); i++) {
            if (i) os << ",";
            os << "{\"x\":" << poly[i].x << ",\"y\":" << poly[i].y << "}";
        }
        os << "]";
    }
}

// "words" is emitted only when non-empty, so the JSON shape is unchanged for
// callers that never set returnWordBoxes — existing wrappers parse this.
void appendWords(std::ostringstream& os, const std::vector<WordBox>& words,
                 bool pretty, int indent) {
    const std::string pad(static_cast<size_t>(indent), ' ');
    const std::string padIn(static_cast<size_t>(indent + 2), ' ');
    os << "[";
    for (size_t i = 0; i < words.size(); i++) {
        if (i) os << ",";
        if (pretty) os << "\n" << padIn;
        os << "{\"text\":\"" << escapeJson(words[i].text) << "\",\"score\":"
           << words[i].score << ",\"polygon\":";
        appendPolygon(os, words[i].polygon, false, 0);
        os << "}";
    }
    if (pretty && !words.empty()) os << "\n" << pad;
    os << "]";
}

void appendLine(std::ostringstream& os, const LinePrediction& line, bool pretty, int indent) {
    const std::string pad(static_cast<size_t>(indent), ' ');
    const std::string padIn(static_cast<size_t>(indent + 2), ' ');
    if (pretty) {
        os << "{\n"
           << padIn << "\"text\":\"" << escapeJson(line.text) << "\",\n"
           << padIn << "\"score\":" << line.score << ",\n"
           << padIn << "\"detScore\":" << line.detScore << ",\n"
           << padIn << "\"polygon\":";
        appendPolygon(os, line.polygon, true, indent + 2);
        if (!line.words.empty()) {
            os << ",\n" << padIn << "\"words\":";
            appendWords(os, line.words, true, indent + 2);
        }
        os << "\n" << pad << "}";
    } else {
        os << "{\"text\":\"" << escapeJson(line.text) << "\",\"score\":" << line.score
           << ",\"detScore\":" << line.detScore << ",\"polygon\":";
        appendPolygon(os, line.polygon, false, 0);
        if (!line.words.empty()) {
            os << ",\"words\":";
            appendWords(os, line.words, false, 0);
        }
        os << "}";
    }
}

} // namespace

std::string toJson(const LinePrediction& line, bool pretty) {
    std::ostringstream os;
    // defaultfloat + 6 significant digits keeps 0.9f as "0.9" (not binary noise).
    os << std::setprecision(6);
    appendLine(os, line, pretty, 0);
    return os.str();
}

std::string toJson(const PagePrediction& page, bool pretty) {
    std::ostringstream os;
    os << std::setprecision(6);
    if (pretty) {
        os << "{\n"
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
        os << "{\"image\":\"" << escapeJson(page.image) << "\",\"elapsedMs\":"
           << page.elapsedMs << ",\"lines\":[";
        for (size_t i = 0; i < page.lines.size(); i++) {
            if (i) os << ",";
            appendLine(os, page.lines[i], false, 0);
        }
        os << "]}";
    }
    return os.str();
}

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

} // namespace arbo::ocr
