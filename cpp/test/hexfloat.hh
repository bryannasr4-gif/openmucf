// The oracle's hexfloat grammar and its canonical renderer -- the C++ half of the rule
// `cpp/tools/build_oracle.py` applies in Python (`HEXFLOAT`, `canonical_hex`, `hexfloat_problem`).
//
// A finite value field must match `^-?0x[01](\.[0-9a-f]{1,13})?p[+-][0-9]+$`, must parse in full,
// and must re-render to the identical bytes under a shortest-hex `%a` reimplementation. Both halves
// are load-bearing: the grammar alone accepts `0x1.50p+3`, `0x1.5p+03` and `0x0.3p+5`, which no
// `%a` ever printed.
//
// The renderer is `frexp`-based rather than `%a` because `%a`'s shape is libc-specific; the parser
// has the same two compiled paths as the reader's decimal parser (`std::from_chars` with
// `chars_format::hex`, or a hand-rolled exact parser where floating-point `from_chars` is absent or
// the stream path is forced).
#ifndef G4MUONICDATA_HEXFLOAT_HH
#define G4MUONICDATA_HEXFLOAT_HH

#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>

#if defined(__cpp_lib_to_chars) && !defined(G4MUONICDATA_FORCE_STREAM_PARSER)
#define G4MUONICDATA_HEX_FROM_CHARS 1
#else
#define G4MUONICDATA_HEX_FROM_CHARS 0
#endif

namespace hexfloat {

inline bool IsLowerHexDigit(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }
inline bool IsDecimalDigit(char c) { return c >= '0' && c <= '9'; }

// `^-?0x[01](\.[0-9a-f]{1,13})?p[+-][0-9]+$`
inline bool MatchesGrammar(const std::string& s) {
  std::size_t i = 0;
  const std::size_t n = s.size();
  if (i < n && s[i] == '-') ++i;
  if (i + 2 > n || s[i] != '0' || s[i + 1] != 'x') return false;
  i += 2;
  if (i >= n || (s[i] != '0' && s[i] != '1')) return false;
  ++i;
  if (i < n && s[i] == '.') {
    ++i;
    std::size_t digits = 0;
    while (i < n && IsLowerHexDigit(s[i])) { ++i; ++digits; }
    if (digits < 1 || digits > 13) return false;
  }
  if (i >= n || s[i] != 'p') return false;
  ++i;
  if (i >= n || (s[i] != '+' && s[i] != '-')) return false;
  ++i;
  std::size_t exponent_digits = 0;
  while (i < n && IsDecimalDigit(s[i])) { ++i; ++exponent_digits; }
  return exponent_digits >= 1 && i == n;
}

inline std::string HexDigits13(std::uint64_t fraction) {
  static const char* const kHex = "0123456789abcdef";
  std::string digits(13, '0');
  for (int i = 12; i >= 0; --i) {
    digits[static_cast<std::size_t>(i)] = kHex[fraction & 0xfu];
    fraction >>= 4;
  }
  while (!digits.empty() && digits.back() == '0') digits.pop_back();
  return digits;
}

// The one spelling the grammar admits: `[-]0x1.<hex, trailing zeros stripped>p<sign><e>` for a
// normal value, `0x0.<hex, trailing zeros stripped>p-1022` for a subnormal, `0x0p+0` / `-0x0p+0`
// for the two zeros. The caller guarantees a finite argument.
inline std::string Canonical(double value) {
  const std::string sign = std::signbit(value) ? "-" : "";
  const double magnitude = std::fabs(value);
  if (magnitude == 0.0) return sign + "0x0p+0";
  if (magnitude < DBL_MIN) {
    // value = fraction * 2^-1074 with fraction < 2^52, exactly.
    const std::uint64_t fraction = static_cast<std::uint64_t>(std::ldexp(magnitude, 1074));
    return sign + "0x0." + HexDigits13(fraction) + "p-1022";
  }
  int exponent = 0;
  const double mantissa = std::frexp(magnitude, &exponent);  // magnitude = mantissa * 2^exponent, mantissa in [0.5, 1)
  const int binary_exponent = exponent - 1;                  // magnitude = (2 * mantissa) * 2^(exponent - 1)
  const std::uint64_t fraction = static_cast<std::uint64_t>(std::ldexp(mantissa * 2.0 - 1.0, 52));  // exact
  const std::string digits = HexDigits13(fraction);
  std::string out = sign + "0x1";
  if (!digits.empty()) out += "." + digits;
  out += "p";
  out += (binary_exponent < 0) ? "-" : "+";
  out += std::to_string(binary_exponent < 0 ? -binary_exponent : binary_exponent);
  return out;
}

// Parse a grammar-conforming field to its double. Returns false when the text is not consumed in
// full (which the grammar should already exclude) or, on the hand-rolled path, when the mantissa
// would not fit 53 bits.
inline bool Parse(const std::string& field, double& out) {
  std::size_t i = 0;
  const bool negative = !field.empty() && field[0] == '-';
  if (negative) ++i;
  if (field.compare(i, 2, "0x") != 0) return false;
  i += 2;
  const std::string body = field.substr(i);  // `[01](.hex)?p[+-]dd`
  double magnitude = 0.0;
#if G4MUONICDATA_HEX_FROM_CHARS
  const char* first = body.data();
  const char* last = first + body.size();
  const std::from_chars_result result = std::from_chars(first, last, magnitude, std::chars_format::hex);
  if (result.ec != std::errc() || result.ptr != last) return false;
#else
  std::size_t j = 0;
  if (j >= body.size() || (body[j] != '0' && body[j] != '1')) return false;
  std::uint64_t mantissa = static_cast<std::uint64_t>(body[j] - '0');
  ++j;
  int fraction_digits = 0;
  if (j < body.size() && body[j] == '.') {
    ++j;
    while (j < body.size() && IsLowerHexDigit(body[j])) {
      const char c = body[j];
      const std::uint64_t nibble = static_cast<std::uint64_t>(IsDecimalDigit(c) ? c - '0' : c - 'a' + 10);
      mantissa = (mantissa << 4) | nibble;
      ++fraction_digits;
      ++j;
    }
    if (fraction_digits == 0 || fraction_digits > 13) return false;  // 1 + 13 * 4 = 53 bits at most
  }
  if (j >= body.size() || body[j] != 'p') return false;
  ++j;
  if (j >= body.size() || (body[j] != '+' && body[j] != '-')) return false;
  const bool exponent_negative = body[j] == '-';
  ++j;
  if (j >= body.size()) return false;
  long exponent = 0;
  while (j < body.size()) {
    if (!IsDecimalDigit(body[j])) return false;
    exponent = exponent * 10 + (body[j] - '0');
    if (exponent > 100000) return false;
    ++j;
  }
  if (exponent_negative) exponent = -exponent;
  // mantissa (an integer below 2^53, exact as a double) times 2^(exponent - 4 * fraction_digits):
  // one std::ldexp, exact for every value the grammar can spell.
  magnitude = std::ldexp(static_cast<double>(mantissa), static_cast<int>(exponent) - 4 * fraction_digits);
#endif
  out = negative ? -magnitude : magnitude;
  return true;
}

// The whole rule: grammar, full parse, canonical re-render. Empty on success, else the reason.
inline std::string Problem(const std::string& field, double& out) {
  if (!MatchesGrammar(field)) {
    return "'" + field + "' is not a canonical hexfloat: a value field is '[-]0x<0|1>[.<up to 13 hex digits>]p<+|-><exponent>', lower case, with the exponent sign always written";
  }
  double value = 0.0;
  if (!Parse(field, value)) return "'" + field + "' did not parse in full as a hexfloat";
  const std::string rendered = Canonical(value);
  if (rendered != field) {
    return "'" + field + "' denotes a value whose canonical spelling is '" + rendered + "'; the harvest prints only canonical spellings";
  }
  out = value;
  return "";
}

}  // namespace hexfloat

#endif
