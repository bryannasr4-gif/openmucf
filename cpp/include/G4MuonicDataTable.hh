// G4MuonicDataTable -- a standard-library-only C++17 reader for the Layer-1 `.g4dat` grammar.
//
// This is the C++ counterpart of `openmucf/g4/spec.py`, the reference implementation of
// FORMAT_SPEC.md sections 2 and 4. It implements the sixteen error codes E001-E016 with the
// three-phase reporting order, and it parses numbers without consulting the process locale
// (FORMAT_SPEC.md section 6). Layer 1 only: the Layer-2 digest (E009) is checked by the validator
// in cpp/test, which has the sibling file; this reader never opens one.
//
// Nothing here depends on Geant4. The class name and the environment variable it resolves are the
// provisional names the dataset documentation uses.
#ifndef G4MUONICDATATABLE_HH
#define G4MUONICDATATABLE_HH

#include <string>
#include <utility>
#include <vector>

class G4MuonicDataTable {
 public:
  // A rejection. `code` is one of E001-E016 for a Layer-1 violation, or empty for a dataset-level
  // problem that has no single line (no `.g4dat` file in the directory, a table name repeated
  // across files, a directory that cannot be read, or an unset discovery variable).
  struct Error {
    std::string code;
    int line = 0;
    std::string text;
    // "{code}: {text} (line {n})" when `code` is non-empty, else `text`.
    std::string what() const;
  };

  struct Table {
    struct Record {
      // The integer columns `Z` and `A` -- whichever the table declares -- in that order
      // (FORMAT_SPEC.md 2.3 rule 6); this is the primary key the records are sorted by.
      std::vector<long> keys;
      // Every other column, in `#COLUMNS` order.
      std::vector<double> floats;
    };
    std::string name;  // the `#TABLE` value
    std::string file;  // the path the table was read from, or the name given to Parse()
    std::vector<std::pair<std::string, std::string>> directives;  // in file order, no leading '#'
    std::vector<std::string> columns;
    std::vector<Record> records;  // ascending by `keys` (E015), unique (E008)
    // The value of the directive with this keyword, or nullptr when the file does not carry it.
    const std::string* Directive(const std::string& keyword) const;
    // Binary search over the sorted records; nullptr when the key is absent.
    const Record* Lookup(const std::vector<long>& key) const;
  };

  // Parse every regular file named `*.g4dat` in `directory`, in bytewise-sorted file-name order.
  // Throws Error: with a code and line for a Layer-1 violation in one file, or code-less when the
  // directory holds no such file, is not a directory, or two files declare the same `#TABLE`.
  static G4MuonicDataTable Load(const std::string& directory);

  // Parse one document from its bytes. `name` is only used to label the resulting table.
  // Throws Error with a code and a 1-based line number on the first violation, in the reporting
  // order of FORMAT_SPEC.md section 4.
  static G4MuonicDataTable Parse(const std::string& bytes, const std::string& name);

  // The table whose `#TABLE` value is `table_name`, or nullptr.
  const Table* Find(const std::string& table_name) const;
  // Every table, in load order (exactly one after Parse()).
  const std::vector<Table>& Tables() const { return tables_; }

  // Overlay opt-in: a plain process-wide flag. Nothing in this reader consults it; it exists so a
  // consumer that falls through to compiled-in values can make that fall-through explicit.
  static void Enable();
  static bool IsEnabled();

  // FORMAT_SPEC.md 2.3 rules 4-5 and section 6, for one float-column field. Returns "" and sets
  // `out` on success; otherwise returns the code ("E007" or "E014") and, when `text` is given,
  // stores the reason in it (without the field, which the caller names). Public because the
  // validator parses `#FALLBACK` coefficients with the same rule the record fields obey.
  static std::string ParseDouble(const std::string& field, double& out, std::string* text = nullptr);

  // Which of the two compiled number-parsing paths this build uses, for the validator's report:
  // "from_chars(__cpp_lib_to_chars=<value>)" or "istringstream(classic)".
  static std::string ParserPath();

 private:
  std::vector<Table> tables_;
};

#endif
