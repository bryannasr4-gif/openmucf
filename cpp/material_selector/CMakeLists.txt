cmake_minimum_required(VERSION 3.16)
project(g4muonic_material_selector LANGUAGES CXX)

# The measured-vector provider and its harness, built against one Geant4 installation.
#   -DG4MUONIC_DRILLS="G4MUONIC_DRILL_X;..."  mutation-drill macros (never in a build that delivers physics)
#   -DG4MUONIC_ASAN=ON                         instrument the provider and the harness with AddressSanitizer
find_package(Geant4 REQUIRED)
set(G4MUONIC_DRILLS "" CACHE STRING "mutation-drill macros")
option(G4MUONIC_ASAN "AddressSanitizer on the provider and the harness" OFF)

# G4HadronicParameters carries the muonic-data switch only in an installation built with cpp/patches/.
set(hadronic_header "")
foreach(inc IN LISTS Geant4_INCLUDE_DIRS)
  if(EXISTS "${inc}/G4HadronicParameters.hh")
    set(hadronic_header "${inc}/G4HadronicParameters.hh")
    set(capture_header "${inc}/G4MuonMinusAtomicCapture.hh")
  endif()
endforeach()
if(NOT hadronic_header)
  message(FATAL_ERROR "G4HadronicParameters.hh was not found")
endif()
file(READ "${hadronic_header}" hadronic_text)

# Does the installation define G4MuonMinusAtomicCapture::SetElementSelector? Link a call to it and see.
set(probe_dir "${CMAKE_BINARY_DIR}/setter_probe")
file(WRITE "${probe_dir}/probe.cc"
  "#include \"G4MuonMinusAtomicCapture.hh\"\nint main(int argc, char**) {\n"
  "  G4MuonMinusAtomicCapture* p = argc > 99 ? new G4MuonMinusAtomicCapture() : nullptr;\n"
  "  if (p) p->SetElementSelector(nullptr);\n  return 0;\n}\n")
try_compile(setter_linked "${probe_dir}/build" SOURCES "${probe_dir}/probe.cc"
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${Geant4_INCLUDE_DIRS}"
  LINK_LIBRARIES ${Geant4_LIBRARIES}
  CXX_STANDARD 17
  OUTPUT_VARIABLE setter_probe_output)
file(WRITE "${CMAKE_BINARY_DIR}/setter_probe.log" "${setter_probe_output}")
file(SHA256 "${capture_header}" capture_sha256)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/supported_headers.txt" supported)
set(header_supported OFF)
foreach(line IN LISTS supported)
  if(line MATCHES " G4MuonMinusAtomicCapture.hh ${capture_sha256}$")
    set(header_supported ON)
  endif()
endforeach()
if(setter_linked)
  set(shim OFF)
  message(STATUS "G4MuonMinusAtomicCapture::SetElementSelector links: no shim")
elseif(header_supported)
  set(shim ON)
  message(STATUS "G4MuonMinusAtomicCapture::SetElementSelector does not link; header ${capture_sha256} is supported: shim")
else()
  message(FATAL_ERROR "SetElementSelector does not link and header ${capture_sha256} is not in supported_headers.txt")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/identity.txt"
  "capture_header ${capture_header} ${capture_sha256}\nsetter_linked ${setter_linked}\nshim ${shim}\n"
  "geant4_version ${Geant4_VERSION}\ncompiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n"
  "drills ${G4MUONIC_DRILLS}\nasan ${G4MUONIC_ASAN}\n")

add_executable(harvest_material harvest_material.cc G4MuonicMaterialSelector.cc ../src/G4MuonicDataTable.cc)
target_include_directories(harvest_material PRIVATE . ../include)
target_compile_features(harvest_material PRIVATE cxx_std_17)
target_compile_options(harvest_material PRIVATE -ffp-contract=off)
target_link_libraries(harvest_material PRIVATE ${Geant4_LIBRARIES})
if(shim)
  target_compile_definitions(harvest_material PRIVATE G4MUONIC_ATOMIC_SETTER_SHIM=1)
endif()
if(hadronic_text MATCHES "SetEnableMuonicData")
  target_compile_definitions(harvest_material PRIVATE G4MUONIC_HARNESS_DATA_SWITCH=1)
endif()
foreach(drill IN LISTS G4MUONIC_DRILLS)
  target_compile_definitions(harvest_material PRIVATE ${drill}=1)
endforeach()
if(G4MUONIC_ASAN)
  target_compile_options(harvest_material PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(harvest_material PRIVATE -fsanitize=address)
endif()
