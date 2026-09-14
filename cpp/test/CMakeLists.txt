# The standalone validator of the G4MuonicData dataset: no Geant4, standard library only.
#
#   cmake -S cpp/test -B build && cmake --build build && ctest --test-dir build --output-on-failure
#
# Options:
#   G4MUONICDATA_FORCE_STREAM_PARSER  build the reader's `std::istringstream` number path even where
#                                     floating-point `std::from_chars` exists (both paths must give
#                                     the same verdicts on the conformance corpus)
#   G4MUONICDATA_PROBE_ONLY           build and register only the toolchain probe
#   G4MUONICDATA_DRILL_CONTRACTION    omit `-ffp-contract=off` from every target, to show the
#                                     contraction detector live (a drill, never a configuration to
#                                     make a parity claim with)
cmake_minimum_required(VERSION 3.16)
project(g4muonicdata_validator LANGUAGES CXX)

# A multi-config generator (Visual Studio, Xcode) chooses the configuration at build time and never
# reads CMAKE_BUILD_TYPE; naming the value here is what stops CMake from reporting it as a
# manually-specified variable the project did not use, on every command line that passes it.
if(CMAKE_CONFIGURATION_TYPES AND DEFINED CMAKE_BUILD_TYPE)
  message(STATUS "multi-config generator: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} is not used; the configuration is chosen at build time with --config")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(G4MUONICDATA_FORCE_STREAM_PARSER "Force the istringstream number-parsing path" OFF)
option(G4MUONICDATA_PROBE_ONLY "Build and register only the toolchain probe" OFF)
option(G4MUONICDATA_DRILL_CONTRACTION "Omit -ffp-contract=off from every target (drill)" OFF)

get_filename_component(G4MUONICDATA_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(G4MUONICDATA_DATASET_DIR "${G4MUONICDATA_REPO_ROOT}/data/g4/d1")
set(G4MUONICDATA_ORACLE "${G4MUONICDATA_DATASET_DIR}/d1_gp_sweep.oracle")
set(G4MUONICDATA_CONFORMANCE_DIR "${G4MUONICDATA_REPO_ROOT}/tests/fixtures/g4dat_conformance")
set(G4MUONICDATA_DOCUMENT "${G4MUONICDATA_REPO_ROOT}/DATASET_D1.md")
set(G4MUONICDATA_TOOLS_README "${G4MUONICDATA_REPO_ROOT}/cpp/tools/README.md")

if(G4MUONICDATA_FORCE_STREAM_PARSER)
  add_compile_definitions(G4MUONICDATA_FORCE_STREAM_PARSER)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(G4MUONICDATA_GNU_LIKE ON)
else()
  set(G4MUONICDATA_GNU_LIKE OFF)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(G4MUONICDATA_X86_64 ON)
else()
  set(G4MUONICDATA_X86_64 OFF)
endif()

# Warning-free on every compiler; contraction off as a TARGET option on GCC/Clang (MSVC gets no
# flag: the runtime detector decides everywhere).
function(g4muonicdata_target_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -pedantic)
    if(NOT G4MUONICDATA_DRILL_CONTRACTION)
      target_compile_options(${target} PRIVATE -ffp-contract=off)
    endif()
  endif()
endfunction()

enable_testing()

# The probe: compiler, __cpp_lib_to_chars, the compiled parser path, the contraction detector.
add_library(g4muonicdatatable STATIC ../src/G4MuonicDataTable.cc)
target_include_directories(g4muonicdatatable PUBLIC ../include)
g4muonicdata_target_options(g4muonicdatatable)

add_executable(g4muonicdata_probe probe.cc)
target_link_libraries(g4muonicdata_probe PRIVATE g4muonicdatatable)
g4muonicdata_target_options(g4muonicdata_probe)
add_test(NAME probe COMMAND g4muonicdata_probe)

if(G4MUONICDATA_PROBE_ONLY)
  return()
endif()

# The hexfloat parser's subnormal path, which no committed oracle field reaches: the grammar's own
# spellings of the subnormal boundary values, parsed, re-rendered and compared bit for bit with a
# reference std::ldexp builds. Standard library only; both parser paths through the FORCE_STREAM
# option.
add_executable(g4muonicdata_hexfloat_check hexfloat_check.cc)
g4muonicdata_target_options(g4muonicdata_hexfloat_check)
add_test(NAME hexfloat_roundtrip COMMAND g4muonicdata_hexfloat_check)

# The conforming GP kernel: the only translation unit that evaluates the model, contraction off.
add_library(gp_kernel_off OBJECT gp_kernel.cc)
target_link_libraries(gp_kernel_off PRIVATE g4muonicdatatable)
target_compile_definitions(gp_kernel_off PRIVATE GP_KERNEL_NS=gp_off)
g4muonicdata_target_options(gp_kernel_off)

add_executable(g4muonicdata_validate validate.cc $<TARGET_OBJECTS:gp_kernel_off>)
target_link_libraries(g4muonicdata_validate PRIVATE g4muonicdatatable)
g4muonicdata_target_options(g4muonicdata_validate)

set(G4MUONICDATA_VALIDATE_ARGS
    "${G4MUONICDATA_DATASET_DIR}" --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")

add_test(NAME validate COMMAND g4muonicdata_validate ${G4MUONICDATA_VALIDATE_ARGS})

# The same run under a comma-decimal locale (Linux only, where the locale can be installed); the
# validator fails the run itself when the locale is absent.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_test(NAME validate_conformance_locale COMMAND g4muonicdata_validate ${G4MUONICDATA_VALIDATE_ARGS})
  set_tests_properties(validate_conformance_locale PROPERTIES ENVIRONMENT "LC_ALL=de_DE.UTF-8")
endif()

# V-12: the three-outcome discovery, four cases, each run with its own environment.
add_test(NAME validate_discovery_found
         COMMAND ${CMAKE_COMMAND} -E env "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}"
                 $<TARGET_FILE:g4muonicdata_validate> ${G4MUONICDATA_VALIDATE_ARGS} --expect-env found)
add_test(NAME validate_discovery_unset
         COMMAND ${CMAKE_COMMAND} -E env --unset=G4MUONICDATA
                 $<TARGET_FILE:g4muonicdata_validate> ${G4MUONICDATA_VALIDATE_ARGS} --expect-env unset)
add_test(NAME validate_discovery_empty
         COMMAND ${CMAKE_COMMAND} -E env "G4MUONICDATA="
                 $<TARGET_FILE:g4muonicdata_validate> ${G4MUONICDATA_VALIDATE_ARGS} --expect-env empty)
# A directory holding one corpus member the reader rejects -- the first rejected row of
# expected.tsv -- copied at configure time; the validator looks its verdict up by file name.
set(G4MUONICDATA_INVALID_MEMBER "")
file(STRINGS "${G4MUONICDATA_CONFORMANCE_DIR}/expected.tsv" G4MUONICDATA_EXPECTED_ROWS)
foreach(row IN LISTS G4MUONICDATA_EXPECTED_ROWS)
  string(REPLACE "\t" ";" parts "${row}")
  list(GET parts 0 member)
  list(GET parts 1 code)
  if(NOT code STREQUAL "OK")
    set(G4MUONICDATA_INVALID_MEMBER "${member}")
    break()
  endif()
endforeach()
if(G4MUONICDATA_INVALID_MEMBER STREQUAL "")
  message(FATAL_ERROR "expected.tsv has no rejected row to build the invalid-discovery case from")
endif()
file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/discovery_invalid")
file(COPY "${G4MUONICDATA_CONFORMANCE_DIR}/${G4MUONICDATA_INVALID_MEMBER}" DESTINATION "${CMAKE_BINARY_DIR}/discovery_invalid")
add_test(NAME validate_discovery_invalid
         COMMAND ${CMAKE_COMMAND} -E env "G4MUONICDATA=${CMAKE_BINARY_DIR}/discovery_invalid"
                 $<TARGET_FILE:g4muonicdata_validate> ${G4MUONICDATA_VALIDATE_ARGS} --expect-env invalid)

# A check over no rows is not a pass: the validator over the shipped dataset, with the conformance
# corpus and the oracle pointed at a fixture that carries zero rows (an empty `expected.tsv`; the
# shipped oracle's header lines only). Every check whose PASS detail counts rows must print FAIL,
# and because each check prints exactly one line the two expressions below say exactly that.
set(G4MUONICDATA_ZERO_ROWS_DIR "${G4MUONICDATA_REPO_ROOT}/tests/fixtures/g4dat_zero_rows")
add_test(NAME validate_zero_rows
         COMMAND g4muonicdata_validate "${G4MUONICDATA_DATASET_DIR}"
                 --oracle "${G4MUONICDATA_ZERO_ROWS_DIR}/d1_gp_sweep.oracle" --conformance "${G4MUONICDATA_ZERO_ROWS_DIR}")
set_tests_properties(validate_zero_rows PROPERTIES
  PASS_REGULAR_EXPRESSION "V-11 FAIL rows=0"
  FAIL_REGULAR_EXPRESSION "V-05 PASS;V-06 PASS;V-08 PASS;V-09 PASS;V-10 PASS;V-11 PASS;V-05 SKIPPED;V-06 SKIPPED;V-08 SKIPPED;V-09 SKIPPED;V-10 SKIPPED;V-11 SKIPPED")

# V-13: the F-3 producer, GCC/Clang only -- the kernel compiled a second time with contraction
# enabled (and FMA on x86-64), and the figures compared with the document.
if(G4MUONICDATA_GNU_LIKE)
  add_library(gp_kernel_fast OBJECT gp_kernel.cc)
  target_link_libraries(gp_kernel_fast PRIVATE g4muonicdatatable)
  target_compile_definitions(gp_kernel_fast PRIVATE GP_KERNEL_NS=gp_fast)
  target_compile_options(gp_kernel_fast PRIVATE -Wall -Wextra -pedantic -ffp-contract=fast)
  if(G4MUONICDATA_X86_64)
    target_compile_options(gp_kernel_fast PRIVATE -mfma)
  endif()

  add_executable(g4muonicdata_f3 f3_producer.cc $<TARGET_OBJECTS:gp_kernel_off> $<TARGET_OBJECTS:gp_kernel_fast>)
  target_link_libraries(g4muonicdata_f3 PRIVATE g4muonicdatatable)
  target_compile_definitions(g4muonicdata_f3 PRIVATE "G4MUONICDATA_CXX_ID=\"${CMAKE_CXX_COMPILER_ID}\"")
  g4muonicdata_target_options(g4muonicdata_f3)

  set(G4MUONICDATA_F3_OUTPUT "${CMAKE_BINARY_DIR}/f3_output.txt")
  add_test(NAME f3_produce
           COMMAND g4muonicdata_f3 "${G4MUONICDATA_DATASET_DIR}" --oracle "${G4MUONICDATA_ORACLE}" --output "${G4MUONICDATA_F3_OUTPUT}")

  set(G4MUONICDATA_F3_CHECK_FLAGS "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND G4MUONICDATA_X86_64)
    list(APPEND G4MUONICDATA_F3_CHECK_FLAGS --exact)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND G4MUONICDATA_X86_64)
    list(APPEND G4MUONICDATA_F3_CHECK_FLAGS --require-fma)
  endif()

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME f3_check
             COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/check_f3.py"
                     --producer-output "${G4MUONICDATA_F3_OUTPUT}" --document "${G4MUONICDATA_DOCUMENT}"
                     --readme "${G4MUONICDATA_TOOLS_README}" ${G4MUONICDATA_F3_CHECK_FLAGS})
    set_tests_properties(f3_check PROPERTIES DEPENDS f3_produce)
  else()
    # Registered as a failure, not omitted: a missing interpreter must not read as a passing check.
    add_test(NAME f3_check_no_python COMMAND ${CMAKE_COMMAND} -E false)
  endif()
endif()
