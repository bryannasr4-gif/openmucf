// harvest_natural.cc -- the natural isotopic composition a Geant4 build gives each element, read from
// the G4Element that G4NistManager builds: the isotopes and relative abundances G4ElementSelector
// reads when it samples the nucleus that captures a stopped negative muon.
//
//   E <Z> <symbol> <listed> <first N> <natural>    for every Z of 1..94: the isotopes the builder
//                                                  lists, its first mass number, and how many it
//                                                  lists with a positive abundance
//   I <Z> <A> <%a>                                 for every isotope of the G4Element built for Z,
//                                                  in the element's order: its mass number and its
//                                                  relative abundance, as the element stores it
//
// An element is built only where the builder lists a positive abundance: building one with none is
// fatal in G4Element. Every value is printed with %a, the exact hexadecimal float. Nothing else is
// written to stdout.
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4NistManager.hh"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace {
constexpr int kFirstZ = 1;
constexpr int kLastZ = 94;

void Harvest() {
  auto* nist = G4NistManager::Instance();
  const auto& symbols = nist->GetNistElementNames();
  for (int z = kFirstZ; z <= kLastZ; ++z) {
    const int listed = nist->GetNumberOfNistIsotopes(z);
    const int first = nist->GetNistFirstIsotopeN(z);
    int natural = 0;
    for (int n = first; n < first + listed; ++n) {
      if (nist->GetIsotopeAbundance(z, n) > 0.) ++natural;
    }
    const std::string symbol = symbols.at(static_cast<std::size_t>(z));
    std::printf("E\t%d\t%s\t%d\t%d\t%d\n", z, symbol.c_str(), listed, first, natural);
    if (natural == 0) continue;
    const G4Element* element = nist->FindOrBuildElement(z);
    if (element == nullptr) throw std::runtime_error("no element built for Z=" + std::to_string(z));
    if (element->GetZasInt() != z || element->GetSymbol() != symbol) {
      throw std::runtime_error("the element built for Z=" + std::to_string(z) + " is not that element");
    }
    const auto* isotopes = element->GetIsotopeVector();
    const auto* abundance = element->GetRelativeAbundanceVector();
    if (static_cast<int>(isotopes->size()) != natural) {
      throw std::runtime_error("the element built for Z=" + std::to_string(z) +
                               " does not carry the isotopes the builder lists with a positive abundance");
    }
    for (std::size_t j = 0; j < isotopes->size(); ++j) {
      const G4Isotope* isotope = (*isotopes)[j];
      if (isotope->GetZ() != z) throw std::runtime_error("an isotope of Z=" + std::to_string(z) + " has another Z");
      std::printf("I\t%d\t%d\t%a\n", z, isotope->GetN(), abundance[j]);
    }
  }
}
}  // namespace

int main() {
  try {
    Harvest();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "harvest_natural: %s\n", error.what());
    return 2;
  }
}
