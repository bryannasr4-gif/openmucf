// The reader's natural-composition rung, called over the two-profile directory the validator's
// V-14 / V-15 case is built from. `LookupNatural` returns the exact record when the table carries
// one; when it does not, and the key has two parts, it falls to the `(Z, 0)` record a table under
// `A:natural_and_listed` carries for that `Z`; and it returns nothing where no such record exists --
// a `Z` without one, and a single-key table (FORMAT_SPEC.md section 6).
//
// Every key below is derived from the loaded tables -- the `(Z, 0)` record found by its key, a mass
// number no record carries built as one more than the largest at that `Z` -- so nothing is typed.
// One `PASS|FAIL <what>` line per assertion; the first FAIL exits non-zero. argv[1] is the directory.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "G4MuonicDataTable.hh"

namespace {

using Table = G4MuonicDataTable::Table;
using Record = Table::Record;

struct Check {
  bool failed = false;
  bool Line(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) failed = true;
    return ok;
  }
};

std::string KeyText(const std::vector<long>& key) {
  std::string text = "(";
  for (std::size_t i = 0; i < key.size(); ++i) text += (i ? ", " : "") + std::to_string(key[i]);
  return text + ")";
}

std::string Where(const Table& t) { return "'#PROFILE " + t.profile + "' / '#TABLE " + t.name + "'"; }

// One more than the largest second key among the records whose first key is `z`: a mass number no
// record at that `Z` carries.
long AbsentA(const Table& t, long z) {
  long max_a = -1;
  for (const Record& r : t.records) {
    if (r.keys.size() == 2 && r.keys[0] == z) max_a = std::max(max_a, r.keys[1]);
  }
  return max_a + 1;
}

bool HasNaturalRow(const Table& t, long z) {
  return t.Lookup({z, 0}) != nullptr;
}

int Run(const char* directory) {
  Check check;
  const G4MuonicDataTable tables = G4MuonicDataTable::Load(directory);
  const Table* evaluated = tables.Find("evaluated", "nuclear_capture_rate");
  const Table* parity = tables.Find(G4MuonicDataTable::kParityProfile, "nuclear_capture_rate");
  const Table* zeff = tables.Find(G4MuonicDataTable::kParityProfile, "muon_zeff");
  if (!check.Line(evaluated != nullptr, "the directory carries '#PROFILE evaluated' / '#TABLE nuclear_capture_rate'")) return 1;
  if (!check.Line(parity != nullptr, "the directory carries '#PROFILE parity' / '#TABLE nuclear_capture_rate'")) return 1;
  if (!check.Line(zeff != nullptr, "the directory carries '#PROFILE parity' / '#TABLE muon_zeff'")) return 1;

  // (i) the evaluated table carries exactly one natural-composition record; a mass number no record
  // at its Z carries, and A = 0 itself, both resolve to that record.
  std::vector<const Record*> natural;
  for (const Record& r : evaluated->records) {
    if (r.keys.size() == 2 && r.keys[1] == 0) natural.push_back(&r);
  }
  if (!check.Line(natural.size() == 1, Where(*evaluated) + " carries " + std::to_string(natural.size()) + " natural-composition record(s), expected exactly one")) return 1;
  const long z0 = natural[0]->keys[0];
  const long absent = AbsentA(*evaluated, z0);
  const Record* exact_natural = evaluated->Lookup({z0, 0});
  if (!check.Line(exact_natural == natural[0], "Lookup" + KeyText({z0, 0}) + " is the natural-composition record")) return 1;
  const Record* fallen = evaluated->LookupNatural({z0, absent});
  if (!check.Line(fallen != nullptr && fallen == exact_natural, "LookupNatural" + KeyText({z0, absent}) + " falls to the record at " + KeyText({z0, 0}))) return 1;
  const Record* at_zero = evaluated->LookupNatural({z0, 0});
  if (!check.Line(at_zero != nullptr && at_zero == exact_natural, "LookupNatural" + KeyText({z0, 0}) + " is the record at " + KeyText({z0, 0}))) return 1;

  // (ii) an exact hit wins on every record of the evaluated table.
  for (const Record& r : evaluated->records) {
    if (!check.Line(evaluated->LookupNatural(r.keys) == &r, "LookupNatural" + KeyText(r.keys) + " on " + Where(*evaluated) + " is the exact record")) return 1;
  }

  // (iii) the first Z of the evaluated table with no (Z, 0) record resolves nothing for an absent A.
  const Record* without_natural = nullptr;
  for (const Record& r : evaluated->records) {
    if (r.keys.size() == 2 && !HasNaturalRow(*evaluated, r.keys[0])) { without_natural = &r; break; }
  }
  if (!check.Line(without_natural != nullptr, Where(*evaluated) + " carries a Z with no natural-composition record")) return 1;
  {
    const long z = without_natural->keys[0];
    const std::vector<long> key{z, AbsentA(*evaluated, z)};
    if (!check.Line(evaluated->LookupNatural(key) == nullptr, "LookupNatural" + KeyText(key) + " on " + Where(*evaluated) + " resolves nothing: no record at " + KeyText({z, 0}))) return 1;
  }

  // (iv) the parity capture table carries no natural-composition record, and an absent A at its
  // first Z resolves nothing.
  std::size_t parity_natural = 0;
  for (const Record& r : parity->records) {
    if (r.keys.size() == 2 && r.keys[1] == 0) ++parity_natural;
  }
  if (!check.Line(parity_natural == 0, Where(*parity) + " carries " + std::to_string(parity_natural) + " natural-composition record(s)")) return 1;
  if (!check.Line(!parity->records.empty() && parity->records.front().keys.size() == 2, Where(*parity) + " carries two-key records")) return 1;
  {
    const long z = parity->records.front().keys[0];
    const std::vector<long> key{z, AbsentA(*parity, z)};
    if (!check.Line(parity->LookupNatural(key) == nullptr, "LookupNatural" + KeyText(key) + " on " + Where(*parity) + " resolves nothing")) return 1;
  }

  // (v) a single-key table is exact only.
  if (!check.Line(!zeff->records.empty() && zeff->records.front().keys.size() == 1, Where(*zeff) + " carries single-key records")) return 1;
  {
    const Record& first = zeff->records.front();
    if (!check.Line(zeff->LookupNatural(first.keys) == &first, "LookupNatural" + KeyText(first.keys) + " on " + Where(*zeff) + " is the exact record")) return 1;
    long max_key = -1;
    for (const Record& r : zeff->records) max_key = std::max(max_key, r.keys[0]);
    const std::vector<long> key{max_key + 1};
    if (!check.Line(zeff->LookupNatural(key) == nullptr, "LookupNatural" + KeyText(key) + " on " + Where(*zeff) + " resolves nothing")) return 1;
  }
  return check.failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("FAIL usage: g4muonicdata_lookup_check <directory>\n");
    return 1;
  }
  try {
    return Run(argv[1]);
  } catch (const G4MuonicDataTable::Error& error) {
    std::printf("FAIL Load: %s\n", error.what().c_str());
    return 1;
  } catch (const std::exception& error) {
    std::printf("FAIL %s\n", error.what());
    return 1;
  }
}
