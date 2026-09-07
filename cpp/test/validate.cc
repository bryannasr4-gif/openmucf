// g4muonicdata_validate -- the standalone validator (no Geant4): checks V-01 .. V-12 over the D1
// dataset directory, the conformance corpus and the committed oracle.
//
//   g4muonicdata_validate <dataset-dir> --oracle <file> --conformance <dir>
//                         [--expect-env found|unset|empty|invalid]
//
// Every check prints exactly one line `V-nn PASS|FAIL|SKIPPED <detail>`; the exit status is 0 iff
// no line says FAIL. A FAIL whose cause is a Layer-1 rejection carries the reader's exact
// `{code}: {text} (line {n})`. A missing or unreadable input path is a named FAIL on the check
// that needed it, never an exception out of main and never a skip.

#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "G4MuonicDataTable.hh"
#include "gp_kernel.hh"
#include "hexfloat.hh"
#include "sha256.hh"

namespace {

using Table = G4MuonicDataTable::Table;
using Error = G4MuonicDataTable::Error;

const char* const kVariable = "G4MUONICDATA";
const char* const kCaptureTable = "nuclear_capture_rate";
const char* const kZeffTable = "muon_zeff";

struct Report {
  bool failed = false;
  void Line(const char* id, const char* verdict, const std::string& detail) {
    std::printf("%s %s %s\n", id, verdict, detail.c_str());
    std::fflush(stdout);
    if (std::strcmp(verdict, "FAIL") == 0) failed = true;
  }
  void Pass(const char* id, const std::string& detail) { Line(id, "PASS", detail); }
  void Fail(const char* id, const std::string& detail) { Line(id, "FAIL", detail); }
  void Skipped(const char* id, const std::string& detail) { Line(id, "SKIPPED", detail); }
};

bool ReadBytes(const std::string& path, std::string& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  out.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return true;
}

std::vector<std::string> SplitWhitespace(const std::string& text) {
  std::vector<std::string> fields;
  std::size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    std::size_t j = i;
    while (j < text.size() && text[j] != ' ' && text[j] != '\t') ++j;
    if (j > i) fields.push_back(text.substr(i, j - i));
    i = j;
  }
  return fields;
}

// Lines of a text; a trailing newline does not add an empty last line.
std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t nl = text.find('\n', start);
    if (nl == std::string::npos) {
      lines.push_back(text.substr(start));
      break;
    }
    lines.push_back(text.substr(start, nl - start));
    start = nl + 1;
  }
  return lines;
}

bool ParseLong(const std::string& s, long& out) {
  if (s.empty()) return false;
  std::size_t i = 0;
  const bool negative = s[0] == '-';
  if (negative) ++i;
  if (i >= s.size()) return false;
  long value = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    value = value * 10 + (s[i] - '0');
    if (value > 100000000L) return false;
  }
  out = negative ? -value : value;
  return true;
}

std::string Quote(const std::string& s) { return "'" + s + "'"; }

bool SameBits(double a, double b) {
  std::uint64_t x = 0, y = 0;
  std::memcpy(&x, &a, sizeof x);
  std::memcpy(&y, &b, sizeof y);
  return x == y;
}

std::int64_t OrderedBits(double x) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &x, sizeof bits);
  return (bits >> 63) ? -static_cast<std::int64_t>(bits & 0x7fffffffffffffffull) : static_cast<std::int64_t>(bits);
}

std::uint64_t UlpDistance(double a, double b) {
  const std::int64_t oa = OrderedBits(a), ob = OrderedBits(b);
  return oa > ob ? static_cast<std::uint64_t>(oa - ob) : static_cast<std::uint64_t>(ob - oa);
}

// Big-endian IEEE-754 binary64 bytes -- the digest rule the oracle header states.
void AppendBigEndian(Sha256& hash, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof bits);
  unsigned char bytes[8];
  for (int i = 0; i < 8; ++i) bytes[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
  hash.Update(bytes, 8);
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// `<stem>.g4dat` -> `<stem>.prov.json`, same directory (the E009 sibling rule).
std::string SiblingPath(const std::string& g4dat_path) {
  const std::string suffix = ".g4dat";
  if (!EndsWith(g4dat_path, suffix)) return "";
  return g4dat_path.substr(0, g4dat_path.size() - suffix.size()) + ".prov.json";
}

std::string FileName(const std::string& path) { return std::filesystem::path(path).filename().string(); }

std::vector<std::string> G4datFilesIn(const std::string& directory, std::string& error) {
  namespace fs = std::filesystem;
  std::vector<std::string> names;
  std::error_code ec;
  for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
    if (fs::is_regular_file(it->path(), ec) && EndsWith(it->path().filename().string(), ".g4dat")) {
      names.push_back(it->path().filename().string());
    }
  }
  if (ec) error = "cannot read directory " + Quote(directory) + ": " + ec.message();
  return names;
}

std::string JoinPath(const std::string& directory, const std::string& name) {
  return (std::filesystem::path(directory) / name).string();
}

// ------------------------------------------------------------------------------------------
// the verdict rule V-05 mirrors: Parse, then the sibling digest (E009 at the #SOURCEDIGEST line)
// ------------------------------------------------------------------------------------------

struct Verdict {
  std::string code = "OK";
  int line = 0;
  std::string what;  // the reader's message; empty when accepted
};

// The #SOURCEDIGEST line of a conforming file: its directives occupy the first lines, in order.
int SourceDigestLine(const Table& table) {
  for (std::size_t i = 0; i < table.directives.size(); ++i) {
    if (table.directives[i].first == "SOURCEDIGEST") return static_cast<int>(i) + 1;
  }
  return 0;
}

// True (and `verdict` set to E009) when the declared digest differs from the sibling's bytes.
bool DigestMismatch(const Table& table, const std::string& sibling_bytes, Verdict& verdict) {
  const std::string* declared = table.Directive("SOURCEDIGEST");
  if (!declared) return false;
  const std::string actual = Sha256Hex(sibling_bytes);
  if (*declared == actual) return false;
  const Error error{"E009", SourceDigestLine(table),
                    "'#SOURCEDIGEST' is " + Quote(*declared) + " but the Layer-2 file hashes to " + Quote(actual)};
  verdict.code = error.code;
  verdict.line = error.line;
  verdict.what = error.what();
  return true;
}

// (code, line) for one `.g4dat` path: a Layer-1 rejection, else E009 when a sibling exists and
// mismatches, else ("OK", 0). False (with `io_error`) when the file cannot be read.
bool VerdictFor(const std::string& g4dat_path, Verdict& verdict, std::string& io_error) {
  std::string bytes;
  if (!ReadBytes(g4dat_path, bytes)) {
    io_error = "cannot read " + Quote(g4dat_path);
    return false;
  }
  verdict = Verdict{};
  try {
    const G4MuonicDataTable parsed = G4MuonicDataTable::Parse(bytes, g4dat_path);
    const std::string sibling = SiblingPath(g4dat_path);
    std::string sibling_bytes;
    if (!sibling.empty() && ReadBytes(sibling, sibling_bytes)) {
      DigestMismatch(parsed.Tables().front(), sibling_bytes, verdict);
    }
  } catch (const Error& error) {
    verdict.code = error.code;
    verdict.line = error.line;
    verdict.what = error.what();
  }
  return true;
}

struct ExpectedRow {
  std::string code;
  int line = 0;
};

// `expected.tsv`: `file<TAB>code<TAB>line`, one row per corpus member.
bool ReadExpected(const std::string& path, std::map<std::string, ExpectedRow>& rows, std::string& error) {
  std::string bytes;
  if (!ReadBytes(path, bytes)) {
    error = "cannot read " + Quote(path);
    return false;
  }
  int number = 0;
  for (const std::string& line : SplitLines(bytes)) {
    ++number;
    const std::size_t t1 = line.find('\t');
    const std::size_t t2 = (t1 == std::string::npos) ? std::string::npos : line.find('\t', t1 + 1);
    long expected_line = 0;
    if (t1 == std::string::npos || t2 == std::string::npos || !ParseLong(line.substr(t2 + 1), expected_line)) {
      error = Quote(path) + " line " + std::to_string(number) + " is not 'file<TAB>code<TAB>line': " + Quote(line);
      return false;
    }
    rows[line.substr(0, t1)] = ExpectedRow{line.substr(t1 + 1, t2 - t1 - 1), static_cast<int>(expected_line)};
  }
  return true;
}

std::string Pair(const std::string& code, int line) { return "(" + code + ", " + std::to_string(line) + ")"; }

// ------------------------------------------------------------------------------------------
// the oracle (V-06 .. V-11)
// ------------------------------------------------------------------------------------------

struct SubsetRow { long z = 0, a = 0; double value = 0.0; int line = 0; };
struct ZeffRow { long z = 0; double value = 0.0; int line = 0; };
struct RateRow { long z = 0, a = 0; std::string classification, field; int line = 0; };

struct Oracle {
  long z_min = 0, z_max = 0, a_min = 0, a_max = 0;
  bool box_read = false;
  std::string fullsweep_sha256;
  std::vector<SubsetRow> subset;
  std::vector<ZeffRow> zeff;
  std::vector<RateRow> rates;
  std::vector<ZeffRow> clamps;
};

bool IsNonFiniteField(const std::string& s) { return s == "nan" || s == "-nan" || s == "inf" || s == "-inf"; }

// The family a non-finite value field belongs to must be the one the classification names.
bool ClassificationAdmits(const std::string& classification, const std::string& field) {
  const std::string family = (field == "nan" || field == "-nan") ? "nan" : "inf";
  std::string cls = classification;
  if (!cls.empty() && (cls[0] == '+' || cls[0] == '-')) cls = cls.substr(1);
  return cls == family;
}

// `# sweep             Z 1..120 x A 1..300 = 36000 points, ...`
bool ParseSweepValue(const std::string& value, Oracle& oracle) {
  const std::size_t z = value.find('Z');
  const std::size_t x = value.find(" x A ");
  const std::size_t eq = value.find(" = ");
  if (z == std::string::npos || x == std::string::npos || eq == std::string::npos || x < z || eq < x) return false;
  auto range = [](const std::string& text, long& lo, long& hi) {
    const std::size_t dots = text.find("..");
    if (dots == std::string::npos) return false;
    return ParseLong(text.substr(0, dots), lo) && ParseLong(text.substr(dots + 2), hi);
  };
  return range(value.substr(z + 2, x - (z + 2)), oracle.z_min, oracle.z_max) &&
         range(value.substr(x + 5, eq - (x + 5)), oracle.a_min, oracle.a_max);
}

// V-06: every row of the oracle against the hexfloat grammar and the re-render rule; returns the
// problems found (empty = the file conforms) and fills `oracle` with the parsed rows.
std::vector<std::string> ReadOracle(const std::string& path, Oracle& oracle) {
  std::vector<std::string> problems;
  std::string bytes;
  if (!ReadBytes(path, bytes)) {
    problems.push_back("cannot read " + Quote(path));
    return problems;
  }
  std::set<std::pair<long, long>> subset_seen, rate_seen;
  std::set<long> zeff_seen, clamp_seen;
  int number = 0;
  bool ended = false;
  for (const std::string& line : SplitLines(bytes)) {
    ++number;
    const std::string where = "line " + std::to_string(number) + ": ";
    if (ended) {
      problems.push_back(where + "content after #END");
      continue;
    }
    if (line == "#END") {
      ended = true;
      continue;
    }
    if (!line.empty() && line[0] == '#') {
      // A header line `# key value` or a bare `#`.
      const std::vector<std::string> words = SplitWhitespace(line.substr(1));
      if (words.size() >= 2 && words[0] == "fullsweep_sha256") oracle.fullsweep_sha256 = words[1];
      if (words.size() >= 2 && words[0] == "sweep") {
        const std::size_t at = line.find("sweep");
        oracle.box_read = ParseSweepValue(line.substr(at + 5), oracle);
        if (!oracle.box_read) problems.push_back(where + "unreadable sweep box " + Quote(line));
      }
      continue;
    }
    const std::vector<std::string> fields = SplitWhitespace(line);
    if (fields.empty()) {
      problems.push_back(where + "blank line inside the oracle");
      continue;
    }
    const std::string& value = fields.back();
    double parsed = 0.0;
    bool finite = true;
    if (IsNonFiniteField(value)) {
      finite = false;
      if (!(fields.size() == 5 && fields[0] == "RATE" && ClassificationAdmits(fields[3], value))) {
        problems.push_back(where + "non-finite field " + Quote(value) + " outside a RATE row whose classification says so");
        continue;
      }
    } else {
      const std::string problem = hexfloat::Problem(value, parsed);
      if (!problem.empty()) {
        problems.push_back(where + problem);
        continue;
      }
    }
    long z = 0, a = 0;
    if (fields.size() == 3 && fields[0] == "ZEFF") {
      if (!ParseLong(fields[1], z)) { problems.push_back(where + "unreadable Z in " + Quote(line)); continue; }
      if (!zeff_seen.insert(z).second) { problems.push_back(where + "duplicate ZEFF row for Z=" + std::to_string(z)); continue; }
      oracle.zeff.push_back(ZeffRow{z, parsed, number});
    } else if (fields.size() == 3 && fields[0] == "ZEFFCLAMP") {
      if (!ParseLong(fields[1], z)) { problems.push_back(where + "unreadable Z in " + Quote(line)); continue; }
      if (!clamp_seen.insert(z).second) { problems.push_back(where + "duplicate ZEFFCLAMP row for Z=" + std::to_string(z)); continue; }
      oracle.clamps.push_back(ZeffRow{z, parsed, number});
    } else if (fields.size() == 5 && fields[0] == "RATE") {
      if (!ParseLong(fields[1], z) || !ParseLong(fields[2], a)) { problems.push_back(where + "unreadable Z A in " + Quote(line)); continue; }
      if (!rate_seen.insert({z, a}).second) { problems.push_back(where + "duplicate RATE row for (Z, A) = (" + std::to_string(z) + ", " + std::to_string(a) + ")"); continue; }
      oracle.rates.push_back(RateRow{z, a, fields[3], value, number});
    } else if (fields.size() == 3 && finite) {
      if (!ParseLong(fields[0], z) || !ParseLong(fields[1], a)) { problems.push_back(where + "unreadable Z A in " + Quote(line)); continue; }
      if (!subset_seen.insert({z, a}).second) { problems.push_back(where + "duplicate sweep row for (Z, A) = (" + std::to_string(z) + ", " + std::to_string(a) + ")"); continue; }
      oracle.subset.push_back(SubsetRow{z, a, parsed, number});
    } else {
      problems.push_back(where + "unrecognised row shape " + Quote(line));
    }
  }
  if (!ended) problems.push_back("no #END line");
  if (!oracle.box_read) problems.push_back("no sweep box in the header");
  if (oracle.fullsweep_sha256.empty()) problems.push_back("no fullsweep_sha256 in the header");
  // Three shape self-checks of the renderer, on computed doubles.
  struct Shape { double value; const char* expected; };
  const Shape shapes[] = {
      {0.0, "0x0p+0"},
      {std::numeric_limits<double>::denorm_min(), "0x0.0000000000001p-1022"},
      {1.0, "0x1p+0"},
  };
  for (const Shape& shape : shapes) {
    const std::string rendered = hexfloat::Canonical(shape.value);
    if (rendered != shape.expected) {
      problems.push_back(std::string("renderer self-check: expected ") + shape.expected + ", rendered " + rendered);
    }
  }
  return problems;
}

// ------------------------------------------------------------------------------------------
// the dataset side: coefficients from #FALLBACK, the zeff table, the value column
// ------------------------------------------------------------------------------------------

struct Model {
  GpCoefficients coefficients;
  std::vector<double> zeff;  // indexed by Z, 0 .. max_z
  long max_z = 0;            // the muon_zeff table's last key
  std::size_t value_index = 0;
  const Table* capture = nullptr;
  const Table* zeff_table = nullptr;
};

// Empty on success, else why the model cannot be built.
std::string BuildModel(const G4MuonicDataTable& tables, Model& model) {
  model.capture = tables.Find(kCaptureTable);
  model.zeff_table = tables.Find(kZeffTable);
  if (!model.capture) return std::string("no table ") + Quote(kCaptureTable);
  if (!model.zeff_table) return std::string("no table ") + Quote(kZeffTable);
  const std::string* fallback = model.capture->Directive("FALLBACK");
  if (!fallback) return "the capture table declares no #FALLBACK";
  const std::vector<std::string> words = SplitWhitespace(*fallback);
  if (words.empty() || words[0] != "goulard_primakoff") return "#FALLBACK model is not goulard_primakoff";
  int seen = 0;
  for (std::size_t k = 1; k < words.size(); ++k) {
    const std::size_t eq = words[k].find('=');
    if (eq == std::string::npos) return "#FALLBACK assignment without '=': " + Quote(words[k]);
    const std::string name = words[k].substr(0, eq), text = words[k].substr(eq + 1);
    double value = 0.0;
    std::string reason;
    const std::string code = G4MuonicDataTable::ParseDouble(text, value, &reason);
    if (!code.empty()) return code + ": " + reason + " in #FALLBACK " + name + "=" + text;
    long integer = 0;
    GpCoefficients& c = model.coefficients;
    if (name == "b0a") c.b0a = value;
    else if (name == "b0b") c.b0b = value;
    else if (name == "b0c") c.b0c = value;
    else if (name == "t1") c.t1 = value;
    else if (name == "xmu_coeff") c.xmu_coeff = value;
    else if (name == "mix") c.mix = value;
    else if (name == "zmin" && ParseLong(text, integer)) c.zmin = integer;
    else if (name == "zmax" && ParseLong(text, integer)) c.zmax = integer;
    else return "unexpected #FALLBACK assignment " + Quote(words[k]);
    ++seen;
  }
  if (seen != 8) return "#FALLBACK declares " + std::to_string(seen) + " assignment(s), the model needs 8";
  {
    std::size_t floats = 0;
    bool found = false;
    for (const std::string& column : model.capture->columns) {
      if (column == "Z" || column == "A") continue;
      if (column == "value") { model.value_index = floats; found = true; }
      ++floats;
    }
    if (!found) return "the capture table has no 'value' column";
  }
  if (model.zeff_table->records.empty()) return "the zeff table is empty";
  model.max_z = model.zeff_table->records.back().keys.front();
  for (long z = 0; z <= model.max_z; ++z) {
    const Table::Record* record = model.zeff_table->Lookup({z});
    if (!record) return "the zeff table has no row for Z=" + std::to_string(z);
    model.zeff.push_back(record->floats.front());
  }
  const GpCoefficients& c = model.coefficients;
  if (!(0 <= c.zmin && c.zmin <= c.zmax && c.zmax <= model.max_z)) {
    return "#FALLBACK clamp [" + std::to_string(c.zmin) + ", " + std::to_string(c.zmax) + "] does not index the zeff table";
  }
  return "";
}

double ClampedZeff(const Model& model, long z) {
  const long zc = z < 1 ? 1 : (z > model.max_z ? model.max_z : z);
  return model.zeff[static_cast<std::size_t>(zc)];
}

GpResult Evaluate(const Model& model, long z, long a) {
  return gp_off::Rate(static_cast<int>(z), static_cast<int>(a), model.coefficients, model.zeff, *model.capture, model.value_index);
}

// ------------------------------------------------------------------------------------------
// discovery: the validator's resolver is std::getenv only (FORMAT_SPEC.md section 5)
// ------------------------------------------------------------------------------------------

struct Resolution {
  int outcome = 0;  // 1 found, 2 unset and unregistered, 3 found but failing to load
  G4MuonicDataTable tables;
  Error error;
};

Resolution Resolve(const char* variable) {
  Resolution resolution;
  const char* value = std::getenv(variable);
  if (value == nullptr) {
    resolution.outcome = 2;
    resolution.error = Error{"", 0,
                             std::string("environment variable ") + variable + " is not set and the dataset is "
                             "unregistered at this Geant4 revision, so GEANT4_DATA_DIR cannot locate it; export " +
                             variable + "=<path to the dataset directory>"};
    return resolution;
  }
  // Any non-null value, the empty string included, is "found" and is handed to Load as it is.
  try {
    resolution.tables = G4MuonicDataTable::Load(value);
    resolution.outcome = 1;
  } catch (const Error& error) {
    resolution.outcome = 3;
    resolution.error = error;
  }
  return resolution;
}

void CheckDiscovery(Report& report, const std::string& expectation, const std::string& dataset,
                    const std::map<std::string, ExpectedRow>& expected, bool expected_loaded) {
  const Resolution r = Resolve(kVariable);
  const char* value = std::getenv(kVariable);
  const std::string seen = "outcome " + std::to_string(r.outcome);
  if (expectation == "found") {
    if (r.outcome != 1) { report.Fail("V-12", "found: expected outcome 1, got " + seen + ": " + r.error.what()); return; }
    if (!r.tables.Find(kCaptureTable) || !r.tables.Find(kZeffTable)) { report.Fail("V-12", "found: outcome 1 but a table is missing"); return; }
    report.Pass("V-12", "found: outcome 1, both tables loaded from " + Quote(value ? value : ""));
    (void)dataset;
    return;
  }
  if (expectation == "unset") {
    if (r.outcome != 2) { report.Fail("V-12", "unset: expected outcome 2, got " + seen); return; }
    const std::string& text = r.error.what();
    if (text.find(kVariable) == std::string::npos || text.find("GEANT4_DATA_DIR") == std::string::npos) {
      report.Fail("V-12", "unset: outcome 2 but the text does not name both variables: " + text);
      return;
    }
    report.Pass("V-12", "unset: outcome 2: " + text);
    return;
  }
  if (expectation == "empty") {
    if (value == nullptr) {
      report.Fail("V-12", std::string("empty: ") + kVariable + " is unset, the case needs it set to the empty string");
      return;
    }
    if (std::strlen(value) != 0) { report.Fail("V-12", std::string("empty: ") + kVariable + " is not empty: " + Quote(value)); return; }
    if (r.outcome != 3 || !r.error.code.empty()) { report.Fail("V-12", "empty: expected outcome 3 with a code-less error, got " + seen + ": " + r.error.what()); return; }
    report.Pass("V-12", "empty: outcome 3 (code-less): " + r.error.what());
    return;
  }
  if (expectation == "invalid") {
    if (!expected_loaded) { report.Fail("V-12", "invalid: expected.tsv was not readable"); return; }
    if (r.outcome != 3) { report.Fail("V-12", "invalid: expected outcome 3, got " + seen); return; }
    if (r.error.code.empty()) { report.Fail("V-12", "invalid: outcome 3 without a code: " + r.error.what()); return; }
    std::string io_error;
    const std::vector<std::string> members = G4datFilesIn(value, io_error);
    if (members.size() != 1) { report.Fail("V-12", "invalid: the directory must hold exactly one .g4dat, found " + std::to_string(members.size()) + (io_error.empty() ? "" : "; " + io_error)); return; }
    const auto row = expected.find(members.front());
    if (row == expected.end()) { report.Fail("V-12", "invalid: no expected.tsv row for " + Quote(members.front())); return; }
    if (row->second.code != r.error.code || row->second.line != r.error.line) {
      report.Fail("V-12", "invalid: " + members.front() + " expected " + Pair(row->second.code, row->second.line) + ", got " + Pair(r.error.code, r.error.line) + ": " + r.error.what());
      return;
    }
    report.Pass("V-12", "invalid: outcome 3, " + members.front() + " " + Pair(r.error.code, r.error.line) + ": " + r.error.what());
    return;
  }
  report.Fail("V-12", "unknown --expect-env value " + Quote(expectation));
}

int Usage() {
  std::fprintf(stderr, "usage: g4muonicdata_validate <dataset-dir> --oracle <file> --conformance <dir> [--expect-env found|unset|empty|invalid]\n");
  return 2;
}

int Run(int argc, char** argv) {
  // First of all: the process locale, so every check below runs under whatever LC_ALL asks for.
  // A requested locale that is not installed is a failure, never a green tick for a run that did
  // not happen the way it was asked to.
  const char* locale = std::setlocale(LC_ALL, "");
  Report report;
  if (locale == nullptr) {
    std::printf("locale=(null)\n");
    report.Fail("V-00", "setlocale(LC_ALL, \"\") returned null: the requested locale is not installed");
  } else {
    std::printf("locale=%s\n", locale);
  }

  std::string dataset, oracle_path, conformance, expectation;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--oracle" && i + 1 < argc) oracle_path = argv[++i];
    else if (arg == "--conformance" && i + 1 < argc) conformance = argv[++i];
    else if (arg == "--expect-env" && i + 1 < argc) expectation = argv[++i];
    else if (!arg.empty() && arg[0] != '-' && dataset.empty()) dataset = arg;
    else return Usage();
  }
  if (dataset.empty() || oracle_path.empty() || conformance.empty()) return Usage();
  if (!expectation.empty() && expectation != "found" && expectation != "unset" && expectation != "empty" && expectation != "invalid") {
    return Usage();
  }

  // V-01 -- the contraction self-test runs before anything else.
  if (ContractionDetected()) {
    report.Fail("V-01", std::string("contracted=YES; ") + ContractionMessage());
    std::printf("%s\n", ContractionMessage());
    return 1;
  }
  report.Pass("V-01", "contracted=NO");

  // V-02 -- the number-parsing path this binary compiled.
  report.Pass("V-02", G4MuonicDataTable::ParserPath());

  // V-03 -- both D1 tables load.
  G4MuonicDataTable tables;
  bool loaded = false;
  try {
    tables = G4MuonicDataTable::Load(dataset);
    const Table* capture = tables.Find(kCaptureTable);
    const Table* zeff = tables.Find(kZeffTable);
    if (!capture || !zeff) {
      report.Fail("V-03", std::string("loaded ") + Quote(dataset) + " but " + (capture ? kZeffTable : kCaptureTable) + " is missing");
    } else {
      loaded = true;
      report.Pass("V-03", std::string(kCaptureTable) + " records=" + std::to_string(capture->records.size()) + " directives=" + std::to_string(capture->directives.size()) +
                              "; " + kZeffTable + " records=" + std::to_string(zeff->records.size()) + " directives=" + std::to_string(zeff->directives.size()));
    }
  } catch (const Error& error) {
    report.Fail("V-03", Quote(dataset) + ": " + error.what());
  }

  // V-04 -- E009 over each loaded table's Layer-2 sibling.
  if (!loaded) {
    report.Fail("V-04", "dataset not loaded");
  } else {
    std::string detail;
    bool ok = true, any_skipped = false;
    for (const char* name : {kCaptureTable, kZeffTable}) {
      const Table* table = tables.Find(name);
      const std::string sibling = SiblingPath(table->file);
      std::string sibling_bytes;
      if (sibling.empty() || !ReadBytes(sibling, sibling_bytes)) {
        report.Skipped("V-04", std::string("(no Layer-2 sibling) ") + std::filesystem::path(table->file).stem().string());
        any_skipped = true;
        continue;
      }
      Verdict verdict;
      if (DigestMismatch(*table, sibling_bytes, verdict)) {
        ok = false;
        detail += (detail.empty() ? "" : "; ") + std::string(name) + ": " + verdict.what;
      } else {
        detail += (detail.empty() ? "" : "; ") + std::string(name) + " sha256(" + FileName(sibling) + ")=" + *table->Directive("SOURCEDIGEST");
      }
    }
    if (!ok) report.Fail("V-04", detail);
    else if (!any_skipped) report.Pass("V-04", detail);
  }

  // V-05 -- the conformance corpus, verdict for verdict.
  std::map<std::string, ExpectedRow> expected;
  std::string expected_error;
  const bool expected_loaded = ReadExpected(JoinPath(conformance, "expected.tsv"), expected, expected_error);
  if (!expected_loaded) {
    report.Fail("V-05", expected_error);
  } else {
    std::string io_error;
    const std::vector<std::string> present = G4datFilesIn(conformance, io_error);
    std::vector<std::string> problems;
    if (!io_error.empty()) problems.push_back(io_error);
    for (const std::string& name : present) {
      if (!expected.count(name)) problems.push_back(name + " is in the directory but has no expected.tsv row");
    }
    std::size_t agreed = 0;
    for (const auto& row : expected) {
      Verdict verdict;
      std::string read_error;
      if (!VerdictFor(JoinPath(conformance, row.first), verdict, read_error)) {
        problems.push_back(row.first + ": " + read_error);
        continue;
      }
      if (verdict.code != row.second.code || verdict.line != row.second.line) {
        problems.push_back(row.first + ": expected " + Pair(row.second.code, row.second.line) + ", got " + Pair(verdict.code, verdict.line) + (verdict.what.empty() ? "" : ": " + verdict.what));
        continue;
      }
      ++agreed;
    }
    if (!problems.empty()) {
      report.Fail("V-05", std::to_string(problems.size()) + " problem(s); first: " + problems.front());
    } else {
      report.Pass("V-05", "rows=" + std::to_string(agreed) + " agree on (code, line)");
    }
  }

  // V-06 -- the oracle grammar.
  Oracle oracle;
  const std::vector<std::string> oracle_problems = ReadOracle(oracle_path, oracle);
  const bool oracle_ok = oracle_problems.empty();
  if (!oracle_ok) {
    report.Fail("V-06", std::to_string(oracle_problems.size()) + " problem(s); first: " + oracle_problems.front());
  } else {
    report.Pass("V-06", "hex fields re-render identically: subset=" + std::to_string(oracle.subset.size()) + " zeff=" + std::to_string(oracle.zeff.size()) +
                            " rate=" + std::to_string(oracle.rates.size()) + " clamp=" + std::to_string(oracle.clamps.size()) + "; renderer shapes verified");
  }

  // The model, for V-07 .. V-11.
  Model model;
  std::string model_error = loaded ? BuildModel(tables, model) : "dataset not loaded";
  const bool model_ok = model_error.empty();
  auto need = [&](const char* id) {
    if (!model_ok) { report.Fail(id, model_error); return false; }
    if (!oracle_ok) { report.Fail(id, "oracle not available: " + oracle_problems.front()); return false; }
    return true;
  };

  // V-07 -- the sweep digest from this binary's own evaluation.
  if (need("V-07")) {
    Sha256 hash;
    std::uint64_t points = 0;
    bool domain_ok = true;
    for (long z = oracle.z_min; z <= oracle.z_max && domain_ok; ++z) {
      for (long a = oracle.a_min; a <= oracle.a_max; ++a) {
        const GpResult r = Evaluate(model, z, a);
        if (!r.ok) {
          report.Fail("V-07", "domain error inside the sweep box at (Z, A) = (" + std::to_string(z) + ", " + std::to_string(a) + ")");
          domain_ok = false;
          break;
        }
        AppendBigEndian(hash, r.value);
        ++points;
      }
    }
    if (domain_ok) {
      const std::string digest = hash.HexDigest();
      if (digest == oracle.fullsweep_sha256) {
        report.Pass("V-07", "points=" + std::to_string(points) + " sha256=" + digest);
      } else {
        report.Fail("V-07", "points=" + std::to_string(points) + " sha256=" + digest + " but the oracle states " + oracle.fullsweep_sha256);
      }
    }
  }

  // V-08 -- every subset row at 0 ulp.
  if (need("V-08")) {
    std::size_t mismatches = 0;
    std::uint64_t worst = 0;
    std::string first;
    for (const SubsetRow& row : oracle.subset) {
      const GpResult r = Evaluate(model, row.z, row.a);
      if (r.ok && SameBits(r.value, row.value)) continue;
      ++mismatches;
      const std::uint64_t ulp = r.ok ? UlpDistance(r.value, row.value) : 0;
      if (ulp > worst) worst = ulp;
      if (first.empty()) {
        first = "line " + std::to_string(row.line) + " (Z, A) = (" + std::to_string(row.z) + ", " + std::to_string(row.a) + ") oracle " +
                hexfloat::Canonical(row.value) + " evaluated " + (r.ok ? hexfloat::Canonical(r.value) : std::string("domain error"));
      }
    }
    if (mismatches) report.Fail("V-08", std::to_string(mismatches) + " row(s) differ, max " + std::to_string(worst) + " ulp; first: " + first);
    else report.Pass("V-08", "rows=" + std::to_string(oracle.subset.size()) + " max_ulp=0");
  }

  // V-09, V-11 -- ZEFF and ZEFFCLAMP rows against the clamp [1, maxZ].
  auto check_clamp = [&](const char* id, const std::vector<ZeffRow>& rows) {
    if (!need(id)) return;
    std::size_t mismatches = 0;
    std::string first;
    for (const ZeffRow& row : rows) {
      const double expected_value = ClampedZeff(model, row.z);
      if (SameBits(expected_value, row.value)) continue;
      ++mismatches;
      if (first.empty()) {
        first = "line " + std::to_string(row.line) + " Z=" + std::to_string(row.z) + " oracle " + hexfloat::Canonical(row.value) + " table " + hexfloat::Canonical(expected_value);
      }
    }
    if (mismatches) report.Fail(id, std::to_string(mismatches) + " row(s) differ; first: " + first);
    else report.Pass(id, "rows=" + std::to_string(rows.size()) + " maxZ=" + std::to_string(model.max_z) + " max_ulp=0");
  };
  check_clamp("V-09", oracle.zeff);

  // V-10 -- the degenerate RATE probes report a domain error; Geant4's behaviour is echoed.
  if (need("V-10")) {
    std::size_t wrong = 0;
    std::string echo;
    for (const RateRow& row : oracle.rates) {
      const GpResult r = Evaluate(model, row.z, row.a);
      if (r.ok) ++wrong;
      echo += " (" + std::to_string(row.z) + "," + std::to_string(row.a) + ")=" + row.classification;
    }
    if (wrong) report.Fail("V-10", std::to_string(wrong) + " probe(s) evaluated instead of reporting a domain error;" + echo);
    else report.Pass("V-10", "probes=" + std::to_string(oracle.rates.size()) + " all report a domain error; Geant4 returns:" + echo);
  }

  check_clamp("V-11", oracle.clamps);

  // V-12 -- discovery, only when asked.
  if (expectation.empty()) {
    report.Skipped("V-12", "(--expect-env not given)");
  } else {
    CheckDiscovery(report, expectation, dataset, expected, expected_loaded);
  }

  return report.failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const Error& error) {
    std::printf("V-XX FAIL unexpected reader error: %s\n", error.what().c_str());
    return 1;
  } catch (const std::exception& error) {
    std::printf("V-XX FAIL unexpected exception: %s\n", error.what());
    return 1;
  }
}
