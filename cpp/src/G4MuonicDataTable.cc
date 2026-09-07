// G4MuonicDataTable -- see the header. Every rule below is stated in FORMAT_SPEC.md and mirrored
// from `openmucf/g4/spec.py`, whose verdicts the conformance corpus records; where the two can be
// read differently, this file follows spec.py.

#include "G4MuonicDataTable.hh"

#include <algorithm>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <locale>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// The two compiled number-parsing paths (FORMAT_SPEC.md section 6). `std::from_chars` for floating
// point is a library feature, so its macro -- not the language standard -- decides; a build can
// force the stream path to exercise it where `from_chars` exists.
#if defined(__cpp_lib_to_chars) && !defined(G4MUONICDATA_FORCE_STREAM_PARSER)
#define G4MUONICDATA_USE_FROM_CHARS 1
#else
#define G4MUONICDATA_USE_FROM_CHARS 0
#endif

namespace {

// FORMAT_SPEC.md 2.2: the one legal directive order. `#END` is not a directive and is not here.
const char* const kDirectiveOrder[] = {
    "GRAMMAR", "DATASET", "VERSION", "PROFILE", "SEAM", "TABLE", "GENERATOR",
    "SOURCEDIGEST", "SOURCESHA", "UNITS", "COLUMNS", "VALIDITY", "FALLBACK",
};
const int kDirectiveCount = static_cast<int>(sizeof(kDirectiveOrder) / sizeof(kDirectiveOrder[0]));
const char* const kAllowedSeams[] = {
    "d1_nuclear_capture", "d2_atomic_capture", "d3_transitions", "d4_mucf_cycle",
};
const char* const kParityProfile = "parity";
const char* const kEndMarker = "#END";
const char* const kSupportedGrammarMajor = "1";
const long kIntegerMin = 0;
const long kIntegerMax = 9999;

using Error = G4MuonicDataTable::Error;
using Table = G4MuonicDataTable::Table;

[[noreturn]] void Fail(const char* code, int line, const std::string& text) {
  throw Error{code, line, text};
}

std::string Quote(const std::string& s) { return "'" + s + "'"; }

int DirectiveIndex(const std::string& keyword) {
  for (int i = 0; i < kDirectiveCount; ++i) {
    if (keyword == kDirectiveOrder[i]) return i;
  }
  return -1;
}

// Required always: every directive except SOURCESHA (required iff parity, E013) and FALLBACK.
bool IsAlwaysRequired(const std::string& keyword) {
  return keyword != "SOURCESHA" && keyword != "FALLBACK";
}

bool IsDigit(char c) { return c >= '0' && c <= '9'; }
bool IsUpper(char c) { return c >= 'A' && c <= 'Z'; }
bool IsLower(char c) { return c >= 'a' && c <= 'z'; }
bool IsLowerHex(char c) { return IsDigit(c) || (c >= 'a' && c <= 'f'); }
bool IsSpaceOrTab(char c) { return c == ' ' || c == '\t'; }

std::string StripSpacesTabs(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && IsSpaceOrTab(s[b])) ++b;
  while (e > b && IsSpaceOrTab(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// FORMAT_SPEC.md 2.3 rule 1: strip both ends, then any run of spaces and tabs is one separator.
std::vector<std::string> SplitFields(const std::string& text) {
  std::vector<std::string> fields;
  const std::string stripped = StripSpacesTabs(text);
  std::size_t i = 0;
  while (i < stripped.size()) {
    std::size_t j = i;
    while (j < stripped.size() && !IsSpaceOrTab(stripped[j])) ++j;
    fields.push_back(stripped.substr(i, j - i));
    while (j < stripped.size() && IsSpaceOrTab(stripped[j])) ++j;
    i = j;
  }
  return fields;
}

// `^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$` -- MAJOR.MINOR without leading zeros (2.7). Returns the
// major as text, or "" when the value does not match.
std::string GrammarMajor(const std::string& value) {
  const std::size_t dot = value.find('.');
  if (dot == std::string::npos) return "";
  const std::string major = value.substr(0, dot), minor = value.substr(dot + 1);
  auto run = [](const std::string& part) {
    if (part.empty()) return false;
    if (part.size() > 1 && part[0] == '0') return false;
    return std::all_of(part.begin(), part.end(), IsDigit);
  };
  return (run(major) && run(minor)) ? major : "";
}

// `^[a-z][a-z0-9_-]{2,31}$` (2.5).
bool IsProfileToken(const std::string& s) {
  if (s.size() < 3 || s.size() > 32 || !IsLower(s[0])) return false;
  return std::all_of(s.begin() + 1, s.end(), [](char c) {
    return IsLower(c) || IsDigit(c) || c == '_' || c == '-';
  });
}

// `^[0-9a-f]{64}$` (2.2).
bool IsDigestShape(const std::string& s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), IsLowerHex);
}

// `^[A-Za-z_][A-Za-z0-9_]*$` (2.2).
bool IsColumnName(const std::string& s) {
  if (s.empty() || !(IsUpper(s[0]) || IsLower(s[0]) || s[0] == '_')) return false;
  return std::all_of(s.begin(), s.end(), [](char c) {
    return IsUpper(c) || IsLower(c) || IsDigit(c) || c == '_';
  });
}

// `^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$` -- the strict C-locale float (2.3 rule 4).
bool MatchesFloatPattern(const std::string& s) {
  std::size_t i = 0;
  const std::size_t n = s.size();
  if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
  std::size_t digits_before = 0;
  while (i < n && IsDigit(s[i])) { ++i; ++digits_before; }
  if (digits_before > 0) {
    if (i < n && s[i] == '.') ++i;
    while (i < n && IsDigit(s[i])) ++i;
  } else {
    if (i >= n || s[i] != '.') return false;
    ++i;
    std::size_t digits_after = 0;
    while (i < n && IsDigit(s[i])) { ++i; ++digits_after; }
    if (digits_after == 0) return false;
  }
  if (i < n && (s[i] == 'e' || s[i] == 'E')) {
    ++i;
    if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
    std::size_t exponent_digits = 0;
    while (i < n && IsDigit(s[i])) { ++i; ++exponent_digits; }
    if (exponent_digits == 0) return false;
  }
  return i == n;
}

std::string Lowercase(std::string s) {
  for (char& c : s) {
    if (IsUpper(c)) c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

// 2.3 rule 5: `inf`, `infinity`, `nan`, any case, with one optional sign.
bool IsNonFiniteToken(const std::string& field) {
  const std::string bare = (!field.empty() && (field[0] == '+' || field[0] == '-')) ? field.substr(1) : field;
  const std::string lower = Lowercase(bare);
  return lower == "inf" || lower == "infinity" || lower == "nan";
}

// A digit 1-9 before the exponent makes the field lexically nonzero (2.3 rule 5).
bool IsLexicallyNonzero(const std::string& field) {
  for (char c : field) {
    if (c == 'e' || c == 'E') break;
    if (c >= '1' && c <= '9') return true;
  }
  return false;
}

// `^#([A-Z][A-Z0-9]*)(?:[ \t]+(.*))?$` -- true with keyword and stripped value on a match.
bool MatchDirective(const std::string& line, std::string& keyword, std::string& value) {
  if (line.empty() || line[0] != '#') return false;
  std::size_t i = 1;
  if (i >= line.size() || !IsUpper(line[i])) return false;
  while (i < line.size() && (IsUpper(line[i]) || IsDigit(line[i]))) ++i;
  keyword = line.substr(1, i - 1);
  if (i == line.size()) {
    value.clear();
    return true;
  }
  if (!IsSpaceOrTab(line[i])) return false;
  while (i < line.size() && IsSpaceOrTab(line[i])) ++i;
  value = StripSpacesTabs(line.substr(i));
  return true;
}

// `^[0-9]+$` within 0-9999 (2.3 rule 3), E007 otherwise; the width question is settled by the bound.
long ParseIntegerField(const std::string& name, const std::string& field, int line) {
  if (field.empty() || !std::all_of(field.begin(), field.end(), IsDigit)) {
    Fail("E007", line, "unreadable unsigned integer in column " + Quote(name) + ": " + Quote(field));
  }
  long value = 0;
  for (char c : field) {
    value = value * 10 + (c - '0');
    if (value > kIntegerMax) break;  // any longer run of digits is out of range as well
  }
  if (value < kIntegerMin || value > kIntegerMax) {
    Fail("E007", line,
         "integer out of range in column " + Quote(name) + ": " + Quote(field) + " is outside " +
             std::to_string(kIntegerMin) + "-" + std::to_string(kIntegerMax));
  }
  return value;
}

std::string PrintedKey(const std::vector<std::string>& key_names, const std::vector<long>& key) {
  std::string out;
  for (std::size_t i = 0; i < key_names.size(); ++i) {
    if (i) out += ", ";
    out += key_names[i] + "=" + std::to_string(key[i]);
  }
  return out;
}

std::string JoinedKeyNames(const std::vector<std::string>& key_names) {
  std::string out;
  for (std::size_t i = 0; i < key_names.size(); ++i) {
    if (i) out += "/";
    out += key_names[i];
  }
  return out;
}

bool g_enabled = false;

}  // namespace

// --------------------------------------------------------------------------------------------
// Error, Table
// --------------------------------------------------------------------------------------------

std::string G4MuonicDataTable::Error::what() const {
  if (code.empty()) return text;
  return code + ": " + text + " (line " + std::to_string(line) + ")";
}

const std::string* G4MuonicDataTable::Table::Directive(const std::string& keyword) const {
  for (const auto& directive : directives) {
    if (directive.first == keyword) return &directive.second;
  }
  return nullptr;
}

const G4MuonicDataTable::Table::Record* G4MuonicDataTable::Table::Lookup(
    const std::vector<long>& key) const {
  auto it = std::lower_bound(records.begin(), records.end(), key,
                             [](const Record& r, const std::vector<long>& k) { return r.keys < k; });
  if (it == records.end() || it->keys != key) return nullptr;
  return &*it;
}

void G4MuonicDataTable::Enable() { g_enabled = true; }
bool G4MuonicDataTable::IsEnabled() { return g_enabled; }

std::string G4MuonicDataTable::ParserPath() {
#if G4MUONICDATA_USE_FROM_CHARS
  return "from_chars(__cpp_lib_to_chars=" + std::to_string(__cpp_lib_to_chars) + ")";
#else
  return "istringstream(classic)";
#endif
}

// --------------------------------------------------------------------------------------------
// numbers -- one function, two compiled paths, the same post-checks (FORMAT_SPEC.md 2.3, 6)
// --------------------------------------------------------------------------------------------

std::string G4MuonicDataTable::ParseDouble(const std::string& field, double& out, std::string* text) {
  auto reject = [&](const char* code, const std::string& why) -> std::string {
    if (text) *text = why;
    return code;
  };
  // Rule 5's non-finite literals are tested BEFORE the pattern: they do not match it, and a reader
  // that applies the pattern first reports E007 where the format requires E014.
  if (IsNonFiniteToken(field)) return reject("E014", "non-finite value");
  if (!MatchesFloatPattern(field)) return reject("E007", "unreadable float");
  // One leading `+` is stripped: the grammar allows it, `std::from_chars` does not.
  const std::string digits = (field[0] == '+') ? field.substr(1) : field;
  double value = 0.0;
#if G4MUONICDATA_USE_FROM_CHARS
  const char* first = digits.data();
  const char* last = first + digits.size();
  const std::from_chars_result result = std::from_chars(first, last, value);
  if (result.ec == std::errc::invalid_argument) return reject("E007", "unreadable float");
  if (result.ec == std::errc::result_out_of_range) {
    // Both ends of the range: overflow to infinity and underflow to zero are E014 alike (2.3 rule
    // 5), and the value is not modified on this outcome, so the code alone decides.
    return reject("E014", "value outside the representable range (overflows to infinity or underflows to zero)");
  }
  if (result.ptr != last) return reject("E007", "unreadable float");
#else
  std::istringstream stream(digits);
  stream.imbue(std::locale::classic());
  stream >> value;
  const bool consumed = stream.eof() || stream.peek() == std::char_traits<char>::eof();
  if (!consumed) return reject("E007", "unreadable float");
  if (stream.fail()) {
    // A library signals overflow and underflow through failbit. Some also flag a SUBNORMAL result
    // that way, and 2.6 guarantees subnormals round-trip, so a finite nonzero subnormal is accepted
    // whatever the flag says; everything else failbit reports is out of range.
    const bool subnormal = std::isfinite(value) && value != 0.0 && std::fabs(value) < DBL_MIN;
    if (!subnormal) {
      return reject("E014", "value outside the representable range (overflows to infinity or underflows to zero)");
    }
  }
#endif
  // Shared post-checks, whichever path produced the value.
  if (std::isinf(value)) return reject("E014", "value overflows to infinity");
  if (value == 0.0 && IsLexicallyNonzero(field)) return reject("E014", "value underflows to zero");
  out = value;
  return "";
}

// --------------------------------------------------------------------------------------------
// Parse -- FORMAT_SPEC.md section 4's three phases, in spec.py's control flow
// --------------------------------------------------------------------------------------------

G4MuonicDataTable G4MuonicDataTable::Parse(const std::string& bytes, const std::string& name) {
  // Phase 1: whole-file lexical. E005 over the bytes, then E006, before any line is looked at.
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(bytes[i]);
    if (c == '\t' || c == '\n' || c == '\r' || (c >= 0x20 && c <= 0x7E)) continue;
    const int line = static_cast<int>(std::count(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(i), '\n')) + 1;
    char hex[8];
    std::snprintf(hex, sizeof hex, "%02X", static_cast<unsigned>(c));
    if (c < 0x80) Fail("E005", line, std::string("control character (0x") + hex + ")");
    Fail("E005", line, std::string("non-ASCII character (U+00") + hex + ")");
  }
  {
    const std::size_t cr = bytes.find('\r');
    if (cr != std::string::npos) {
      const int line = static_cast<int>(std::count(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(cr), '\n')) + 1;
      const bool crlf = cr + 1 < bytes.size() && bytes[cr + 1] == '\n';
      Fail("E006", line, std::string(crlf ? "CRLF" : "CR") + " line ending; this format is LF-only");
    }
  }

  // `text.split("\n")`: a trailing newline yields one empty last element.
  std::vector<std::string> raw;
  {
    std::size_t start = 0;
    while (true) {
      const std::size_t nl = bytes.find('\n', start);
      if (nl == std::string::npos) {
        raw.push_back(bytes.substr(start));
        break;
      }
      raw.push_back(bytes.substr(start, nl - start));
      start = nl + 1;
    }
  }
  const bool newline_terminated = !raw.empty() && raw.back().empty();
  const int body_size = static_cast<int>(raw.size()) - (newline_terminated ? 1 : 0);

  Table table;
  table.file = name;
  std::map<std::string, int> directive_lines;
  auto has_directive = [&](const std::string& keyword) { return directive_lines.count(keyword) != 0; };
  auto directive_value = [&](const std::string& keyword) -> const std::string& {
    return *table.Directive(keyword);
  };
  std::vector<int> record_lines;
  bool columns_fixed = false;
  int order_index = -1;
  int end_line = 0;

  // A present directive is reported at its own line; a missing one at the line it would have
  // occupied: the count of directives that precede it in the legal order and are present, plus one.
  auto dline = [&](const std::string& keyword) -> int {
    auto found = directive_lines.find(keyword);
    if (found != directive_lines.end()) return found->second;
    int preceding = 0;
    for (int i = 0; i < DirectiveIndex(keyword); ++i) {
      if (has_directive(kDirectiveOrder[i])) ++preceding;
    }
    return preceding + 1;
  };

  // Phase 3's block-close group: all E002 in directive order, then E016 for PROFILE, SEAM,
  // SOURCEDIGEST, COLUMNS, then E013. (E010 cannot fire here: it was checked eagerly.)
  auto close_header = [&]() {
    for (int i = 0; i < kDirectiveCount; ++i) {
      const std::string keyword = kDirectiveOrder[i];
      if (!IsAlwaysRequired(keyword)) continue;
      if (!has_directive(keyword)) {
        Fail("E002", dline(keyword), "missing required directive '#" + keyword + "'");
      }
      if (directive_value(keyword).empty()) {
        Fail("E002", dline(keyword),
             "required directive '#" + keyword + "' has an empty value, which counts as absent");
      }
    }
    const std::string& profile = directive_value("PROFILE");
    if (!IsProfileToken(profile)) {
      Fail("E016", dline("PROFILE"),
           "'#PROFILE' value " + Quote(profile) + " is not a ^[a-z][a-z0-9_-]{2,31}$ token");
    }
    const std::string& seam = directive_value("SEAM");
    if (std::find(std::begin(kAllowedSeams), std::end(kAllowedSeams), seam) == std::end(kAllowedSeams)) {
      std::string allowed;
      for (const char* s : kAllowedSeams) allowed += (allowed.empty() ? "" : ", ") + std::string(s);
      Fail("E016", dline("SEAM"), "'#SEAM' value " + Quote(seam) + " is not one of " + allowed);
    }
    const std::string& digest = directive_value("SOURCEDIGEST");
    if (!IsDigestShape(digest)) {
      Fail("E016", dline("SOURCEDIGEST"),
           "'#SOURCEDIGEST' value " + Quote(digest) + " is not 64 lowercase hex characters");
    }
    const std::vector<std::string> columns = SplitFields(directive_value("COLUMNS"));
    for (const std::string& column : columns) {
      if (!IsColumnName(column)) {
        Fail("E016", dline("COLUMNS"),
             "'#COLUMNS' name " + Quote(column) + " is not a ^[A-Za-z_][A-Za-z0-9_]*$ token");
      }
    }
    {
      std::vector<std::string> sorted = columns;
      std::sort(sorted.begin(), sorted.end());
      std::string duplicates;
      for (std::size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] == sorted[i - 1] && (i < 2 || sorted[i] != sorted[i - 2])) {
          duplicates += (duplicates.empty() ? "" : ", ") + sorted[i];
        }
      }
      if (!duplicates.empty()) {
        Fail("E016", dline("COLUMNS"), "'#COLUMNS' repeats the name(s) " + duplicates);
      }
    }
    const bool parity = profile == kParityProfile;
    const bool has_sha = has_directive("SOURCESHA") && !directive_value("SOURCESHA").empty();
    if (parity && !has_sha) {
      Fail("E013", dline("PROFILE"),
           std::string("'#PROFILE ") + kParityProfile + "' requires a non-empty '#SOURCESHA'");
    }
    if (!parity && has_sha) {
      Fail("E013", dline("SOURCESHA"),
           std::string("'#SOURCESHA' is only allowed under '#PROFILE ") + kParityProfile + "', not " + Quote(profile));
    }
    table.columns = columns;
    columns_fixed = true;
  };

  // Phase 2: the line scan, in file order.
  for (int lineno = 1; lineno <= body_size; ++lineno) {
    const std::string& line = raw[static_cast<std::size_t>(lineno - 1)];
    if (end_line != 0) {
      Fail("E011", lineno, std::string("content after '") + kEndMarker + "': " + Quote(StripSpacesTabs(line)));
    }
    {
      // The terminator: `#END` with only spaces and tabs after it.
      std::size_t e = line.size();
      while (e > 0 && IsSpaceOrTab(line[e - 1])) --e;
      if (line.substr(0, e) == kEndMarker) {
        if (!columns_fixed) close_header();
        end_line = lineno;
        continue;
      }
    }
    if (!line.empty() && line[0] == '#') {
      std::string keyword, value;
      const bool matched = MatchDirective(line, keyword, value);
      // A terminator carrying content is E011 in every position; `END` is not a directive.
      if (matched && keyword == "END") {
        Fail("E011", lineno, std::string("the '") + kEndMarker + "' terminator line carries content: " + Quote(value));
      }
      if (columns_fixed) {
        const std::vector<std::string> words = SplitFields(line);
        Fail("E003", lineno, "directive " + Quote(words.empty() ? line : words[0]) + " appears after the record block began");
      }
      if (!matched) Fail("E001", lineno, "unreadable directive line " + Quote(StripSpacesTabs(line)));
      const int index = DirectiveIndex(keyword);
      if (index < 0) Fail("E001", lineno, "unknown directive '#" + keyword + "'");
      if (index <= order_index) {
        const std::string detail = has_directive(keyword)
                                       ? "repeated"
                                       : "must precede '#" + std::string(kDirectiveOrder[order_index]) + "'";
        Fail("E003", lineno, "directive '#" + keyword + "' is out of order (" + detail + ")");
      }
      order_index = index;
      table.directives.emplace_back(keyword, value);
      directive_lines[keyword] = lineno;
      if (keyword == "GRAMMAR") {
        // Eagerly, at its own line: E002 for an empty value, E010 for a version this reader cannot
        // read; either preempts everything after it (2.7).
        if (value.empty()) {
          Fail("E002", lineno, "required directive '#GRAMMAR' has an empty value, which counts as absent");
        }
        const std::string major = GrammarMajor(value);
        if (major.empty()) {
          Fail("E010", lineno, "unreadable '#GRAMMAR' version " + Quote(value) + "; expected MAJOR.MINOR without leading zeros");
        }
        if (major != kSupportedGrammarMajor) {
          Fail("E010", lineno, "unsupported '#GRAMMAR' major version " + major + "; this reader implements major " + kSupportedGrammarMajor);
        }
      }
      continue;
    }

    // A record line. The first one closes the directive block, and the block-close rules run at
    // that line BEFORE it is read as a record (the same-line tie-break of section 4).
    if (!columns_fixed) close_header();
    const std::vector<std::string> fields = SplitFields(line);
    if (fields.size() != table.columns.size()) {
      Fail("E004", lineno,
           "record has " + std::to_string(fields.size()) + " field(s), '#COLUMNS' declares " + std::to_string(table.columns.size()));
    }
    Table::Record record;
    std::vector<long> z_a(2, 0);
    bool has_z = false, has_a = false;
    for (std::size_t i = 0; i < fields.size(); ++i) {
      const std::string& column = table.columns[i];
      if (column == "Z" || column == "A") {
        const long integer = ParseIntegerField(column, fields[i], lineno);
        if (column == "Z") { z_a[0] = integer; has_z = true; } else { z_a[1] = integer; has_a = true; }
        continue;
      }
      double value = 0.0;
      std::string reason;
      const std::string code = ParseDouble(fields[i], value, &reason);
      if (!code.empty()) {
        Fail(code.c_str(), lineno, reason + " in column " + Quote(column) + ": " + Quote(fields[i]));
      }
      record.floats.push_back(value);
    }
    if (has_z) record.keys.push_back(z_a[0]);
    if (has_a) record.keys.push_back(z_a[1]);
    table.records.push_back(record);
    record_lines.push_back(lineno);
  }

  if (end_line == 0) {
    Fail("E012", std::max(body_size, 1), std::string("missing '") + kEndMarker + "' terminator");
  }
  if (!newline_terminated) {
    Fail("E012", body_size, std::string("the '") + kEndMarker + "' line is not newline-terminated; the file must end with a newline");
  }

  // Phase 3, once all records are in hand: E008 across all records, then E015.
  std::vector<std::string> key_names;
  for (const char* k : {"Z", "A"}) {
    if (std::find(table.columns.begin(), table.columns.end(), k) != table.columns.end()) key_names.push_back(k);
  }
  if (!key_names.empty()) {
    std::map<std::vector<long>, int> first_seen;
    for (std::size_t i = 0; i < table.records.size(); ++i) {
      const std::vector<long>& key = table.records[i].keys;
      auto seen = first_seen.find(key);
      if (seen != first_seen.end()) {
        Fail("E008", record_lines[i],
             "duplicate key (" + PrintedKey(key_names, key) + "); first seen at line " + std::to_string(seen->second));
      }
      first_seen[key] = record_lines[i];
    }
    for (std::size_t i = 1; i < table.records.size(); ++i) {
      if (table.records[i].keys < table.records[i - 1].keys) {
        Fail("E015", record_lines[i],
             "record (" + PrintedKey(key_names, table.records[i].keys) + ") is not in ascending " + JoinedKeyNames(key_names) + " order");
      }
    }
  }

  if (const std::string* table_name = table.Directive("TABLE")) table.name = *table_name;
  G4MuonicDataTable result;
  result.tables_.push_back(std::move(table));
  return result;
}

// --------------------------------------------------------------------------------------------
// Load, Find
// --------------------------------------------------------------------------------------------

G4MuonicDataTable G4MuonicDataTable::Load(const std::string& directory) {
  namespace fs = std::filesystem;
  std::vector<fs::path> files;
  std::error_code ec;
  if (!fs::is_directory(directory, ec)) {
    throw Error{"", 0, "dataset directory " + Quote(directory) + " is not a readable directory"};
  }
  for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
    const fs::path& path = it->path();
    if (!fs::is_regular_file(path, ec)) continue;
    const std::string filename = path.filename().string();
    const std::string suffix = ".g4dat";
    if (filename.size() > suffix.size() && filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
      files.push_back(path);
    }
  }
  if (ec) {
    throw Error{"", 0, "dataset directory " + Quote(directory) + " cannot be read: " + ec.message()};
  }
  if (files.empty()) {
    throw Error{"", 0, "dataset directory " + Quote(directory) + " holds no *.g4dat file"};
  }
  std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
    return a.filename().string() < b.filename().string();
  });
  G4MuonicDataTable result;
  std::map<std::string, std::string> table_files;
  for (const fs::path& path : files) {
    // Binary mode, always: text mode on Windows would strip CR before the E006 check sees it.
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error{"", 0, "cannot open " + Quote(path.string())};
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    G4MuonicDataTable one = Parse(bytes, path.string());
    Table& table = one.tables_.front();
    auto seen = table_files.find(table.name);
    if (seen != table_files.end()) {
      throw Error{"", 0, "'#TABLE " + table.name + "' is declared by both " + Quote(seen->second) + " and " + Quote(path.string())};
    }
    table_files[table.name] = path.string();
    result.tables_.push_back(std::move(table));
  }
  return result;
}

const G4MuonicDataTable::Table* G4MuonicDataTable::Find(const std::string& table_name) const {
  for (const Table& table : tables_) {
    if (table.name == table_name) return &table;
  }
  return nullptr;
}
