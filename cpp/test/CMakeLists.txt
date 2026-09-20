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
set(G4MUONICDATA_D1_DIR "${G4MUONICDATA_REPO_ROOT}/data/g4/d1")
set(G4MUONICDATA_D3_DIR "${G4MUONICDATA_REPO_ROOT}/data/g4/d3")
set(G4MUONICDATA_ORACLE "${G4MUONICDATA_D1_DIR}/d1_gp_sweep.oracle")
# The dataset the validator reads: every Layer-1 and Layer-2 member of the D1 and D3 directories,
# copied into one directory at configure time, as the archive installs them side by side.
set(G4MUONICDATA_DATASET_DIR "${CMAKE_BINARY_DIR}/dataset")
file(REMOVE_RECURSE "${G4MUONICDATA_DATASET_DIR}")
file(GLOB G4MUONICDATA_DATASET_MEMBERS
     "${G4MUONICDATA_D1_DIR}/*.g4dat" "${G4MUONICDATA_D1_DIR}/*.prov.json"
     "${G4MUONICDATA_D3_DIR}/*.g4dat" "${G4MUONICDATA_D3_DIR}/*.prov.json")
file(COPY ${G4MUONICDATA_DATASET_MEMBERS} DESTINATION "${G4MUONICDATA_DATASET_DIR}")
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
add_library(g4muonicdatatable STATIC ../src/G4MuonicDataTable.cc ../src/G4MuonicDataSemantics.cc)
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

# One oracle section empty at a time: each fixture carries the shipped header, the first row of
# every OTHER section, and `#END`, so V-06's FAIL names exactly the one empty section (the
# trailing newline in the expression is what excludes a second name).
foreach(section IN ITEMS subset zeff rate clamp)
  add_test(NAME validate_zero_rows_${section}
           COMMAND g4muonicdata_validate "${G4MUONICDATA_DATASET_DIR}"
                   --oracle "${G4MUONICDATA_ZERO_ROWS_DIR}/zero_${section}.oracle" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
  set_tests_properties(validate_zero_rows_${section} PROPERTIES
    PASS_REGULAR_EXPRESSION "V-06 FAIL rows=0: a check over no rows is not a pass: ${section}\n")
endforeach()

# V-03's pair rule, V-14 and V-15 over directories assembled at configure time from the shipped
# dataset and corpus members copied under the name the case needs (configure_file COPYONLY copies
# the bytes; file(COPY) keeps the shipped names).
set(G4MUONICDATA_SHIPPED_FILES
    "${G4MUONICDATA_D1_DIR}/d1_capture.g4dat" "${G4MUONICDATA_D1_DIR}/d1_capture.prov.json"
    "${G4MUONICDATA_D1_DIR}/d1_zeff.g4dat" "${G4MUONICDATA_D1_DIR}/d1_zeff.prov.json")

# A second profile of the capture table, carrying one natural-composition row under
# A:natural_and_listed: both tables of parity plus the evaluated file load, and both checks pass.
set(G4MUONICDATA_PROFILES_TWO "${CMAKE_BINARY_DIR}/profiles_two")
file(REMOVE_RECURSE "${G4MUONICDATA_PROFILES_TWO}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} DESTINATION "${G4MUONICDATA_PROFILES_TWO}")
configure_file("${G4MUONICDATA_CONFORMANCE_DIR}/ok_natural_row_under_natural_and_listed.g4dat"
               "${G4MUONICDATA_PROFILES_TWO}/d1_capture.evaluated.g4dat" COPYONLY)
add_test(NAME validate_two_profiles
         COMMAND g4muonicdata_validate "${G4MUONICDATA_PROFILES_TWO}"
                 --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_two_profiles PROPERTIES
  PASS_REGULAR_EXPRESSION "V-14 PASS profiles=evaluated,parity pairs=3\nV-15 PASS tables=3 natural_rows=1\n")

# The reader's natural-composition rung, called over the same directory: an exact hit wins, a mass
# number no record carries falls to the (Z, 0) record under A:natural_and_listed, and a Z without
# one resolves nothing. Every key is derived from
# the loaded tables.
add_executable(g4muonicdata_lookup_check lookup_check.cc)
target_link_libraries(g4muonicdata_lookup_check PRIVATE g4muonicdatatable)
g4muonicdata_target_options(g4muonicdata_lookup_check)
add_test(NAME lookup_natural COMMAND g4muonicdata_lookup_check "${G4MUONICDATA_PROFILES_TWO}")

# Two files declaring the same (#PROFILE, #TABLE) pair: the shipped capture file under a second
# name. The reader refuses the directory and names both files.
set(G4MUONICDATA_PROFILES_SAME_PAIR "${CMAKE_BINARY_DIR}/profiles_same_pair")
file(REMOVE_RECURSE "${G4MUONICDATA_PROFILES_SAME_PAIR}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} DESTINATION "${G4MUONICDATA_PROFILES_SAME_PAIR}")
configure_file("${G4MUONICDATA_D1_DIR}/d1_capture.g4dat"
               "${G4MUONICDATA_PROFILES_SAME_PAIR}/d1_capture_copy.g4dat" COPYONLY)
add_test(NAME validate_same_pair
         COMMAND g4muonicdata_validate "${G4MUONICDATA_PROFILES_SAME_PAIR}"
                 --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_same_pair PROPERTIES
  PASS_REGULAR_EXPRESSION "the pair '#PROFILE parity' / '#TABLE nuclear_capture_rate' is declared by both")

# A natural-composition row under A:listed, in a second profile: V-15 names the range.
set(G4MUONICDATA_PROFILES_NATURAL_UNDER_LISTED "${CMAKE_BINARY_DIR}/profiles_natural_under_listed")
file(REMOVE_RECURSE "${G4MUONICDATA_PROFILES_NATURAL_UNDER_LISTED}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} DESTINATION "${G4MUONICDATA_PROFILES_NATURAL_UNDER_LISTED}")
configure_file("${G4MUONICDATA_CONFORMANCE_DIR}/ok_natural_row_under_listed.g4dat"
               "${G4MUONICDATA_PROFILES_NATURAL_UNDER_LISTED}/d1_capture.evaluated.g4dat" COPYONLY)
add_test(NAME validate_natural_row_under_listed
         COMMAND g4muonicdata_validate "${G4MUONICDATA_PROFILES_NATURAL_UNDER_LISTED}"
                 --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_natural_row_under_listed PROPERTIES
  PASS_REGULAR_EXPRESSION "V-15 FAIL under A:listed")

# A natural-composition row in the parity capture table itself (the corpus member as the capture
# file, beside the shipped zeff pair): V-15 refuses it under parity whatever #VALIDITY says.
set(G4MUONICDATA_PROFILES_NATURAL_IN_PARITY "${CMAKE_BINARY_DIR}/profiles_natural_in_parity")
file(REMOVE_RECURSE "${G4MUONICDATA_PROFILES_NATURAL_IN_PARITY}")
file(COPY "${G4MUONICDATA_D1_DIR}/d1_zeff.g4dat" "${G4MUONICDATA_D1_DIR}/d1_zeff.prov.json"
     DESTINATION "${G4MUONICDATA_PROFILES_NATURAL_IN_PARITY}")
configure_file("${G4MUONICDATA_CONFORMANCE_DIR}/ok_natural_row_under_parity.g4dat"
               "${G4MUONICDATA_PROFILES_NATURAL_IN_PARITY}/d1_capture.g4dat" COPYONLY)
add_test(NAME validate_natural_row_in_parity
         COMMAND g4muonicdata_validate "${G4MUONICDATA_PROFILES_NATURAL_IN_PARITY}"
                 --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_natural_row_in_parity PROPERTIES
  PASS_REGULAR_EXPRESSION "V-15 FAIL under parity")

# A second profile carrying a table parity does not: the shipped pair plus the format example,
# whose (#PROFILE evaluated, #TABLE format_example) has no parity partner. V-14 names the table
# and the file.
set(G4MUONICDATA_PROFILE_WITHOUT_PARITY "${CMAKE_BINARY_DIR}/profile_without_parity")
file(REMOVE_RECURSE "${G4MUONICDATA_PROFILE_WITHOUT_PARITY}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} "${G4MUONICDATA_REPO_ROOT}/data/g4/example.g4dat"
     DESTINATION "${G4MUONICDATA_PROFILE_WITHOUT_PARITY}")
add_test(NAME validate_profile_without_parity
         COMMAND g4muonicdata_validate "${G4MUONICDATA_PROFILE_WITHOUT_PARITY}"
                 --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_profile_without_parity PROPERTIES
  PASS_REGULAR_EXPRESSION "V-14 FAIL '#PROFILE evaluated' carries '#TABLE format_example' [(]example.g4dat[)] that parity does not\n")

# The #VALIDITY sub-grammar's two refusals. Each directory is derived at configure time from the
# two-profile case above: the shipped pair, plus the corpus member with its `A:natural_and_listed`
# assignment replaced by the text the case needs -- one stated replacement of a committed member,
# so no second copy of the member is committed.
function(g4muonicdata_validity_case name replacement expected)
  set(dir "${CMAKE_BINARY_DIR}/${name}")
  file(REMOVE_RECURSE "${dir}")
  file(COPY ${G4MUONICDATA_SHIPPED_FILES} DESTINATION "${dir}")
  file(READ "${G4MUONICDATA_CONFORMANCE_DIR}/ok_natural_row_under_natural_and_listed.g4dat" member)
  string(FIND "${member}" "A:natural_and_listed" at)
  if(at EQUAL -1)
    message(FATAL_ERROR "ok_natural_row_under_natural_and_listed.g4dat does not assign A:natural_and_listed")
  endif()
  string(REPLACE "A:natural_and_listed" "${replacement}" member "${member}")
  # file(WRITE) opens its output in text mode and writes CRLF on Windows, which the LF-only format
  # refuses (E006); configure_file with NEWLINE_STYLE LF writes LF on every platform. @ONLY leaves
  # the bytes alone provided the member carries no @ and no #cmakedefine, which is checked here.
  foreach(marker IN ITEMS "@" "#cmakedefine")
    string(FIND "${member}" "${marker}" at)
    if(NOT at EQUAL -1)
      message(FATAL_ERROR "ok_natural_row_under_natural_and_listed.g4dat carries '${marker}', which configure_file would rewrite")
    endif()
  endforeach()
  file(WRITE "${CMAKE_BINARY_DIR}/${name}.member.in" "${member}")
  configure_file("${CMAKE_BINARY_DIR}/${name}.member.in" "${dir}/d1_capture.evaluated.g4dat"
                 @ONLY NEWLINE_STYLE LF)
  add_test(NAME ${name}
           COMMAND g4muonicdata_validate "${dir}"
                   --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
  set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${expected}")
endfunction()
g4muonicdata_validity_case(validate_validity_without_range "A"
  "V-15 FAIL 'd1_capture.evaluated.g4dat' #VALIDITY assignment 'A' is not NAME:RANGE\n")
g4muonicdata_validity_case(validate_validity_assigned_twice "A:listed A:natural_and_listed"
  "V-15 FAIL 'd1_capture.evaluated.g4dat' #VALIDITY assigns 'A' twice\n")

# The D3 energy tables of the mudirac130 profile: V-14 over a seam parity carries no table in, and
# V-15 and V-16 over the two tables. Every directory below is assembled at configure time from the
# shipped D1 pair and the shipped D3 files; a changed member is one stated edit of a shipped file.
set(G4MUONICDATA_D3_KSHELL "${G4MUONICDATA_D3_DIR}/d3_kshell.mudirac130.g4dat")
set(G4MUONICDATA_D3_LEVELS "${G4MUONICDATA_D3_DIR}/d3_levels.mudirac130.g4dat")
foreach(member IN ITEMS "${G4MUONICDATA_D3_KSHELL}" "${G4MUONICDATA_D3_LEVELS}")
  if(NOT EXISTS "${member}")
    message(FATAL_ERROR "the D3 case needs ${member}, which is not in the tree")
  endif()
endforeach()

# Write `text` as `name` in `dir`, LF on every platform (see g4muonicdata_validity_case above).
function(g4muonicdata_write_member dir name text)
  foreach(marker IN ITEMS "@" "#cmakedefine")
    string(FIND "${text}" "${marker}" at)
    if(NOT at EQUAL -1)
      message(FATAL_ERROR "${name} carries '${marker}', which configure_file would rewrite")
    endif()
  endforeach()
  file(WRITE "${CMAKE_BINARY_DIR}/${name}.in" "${text}")
  configure_file("${CMAKE_BINARY_DIR}/${name}.in" "${dir}/${name}" @ONLY NEWLINE_STYLE LF)
endfunction()

# The full shipped set: every profile, V-14 naming all three, V-16 passing over mudirac130.
add_test(NAME validate_d3_profiles COMMAND g4muonicdata_validate ${G4MUONICDATA_VALIDATE_ARGS})
set_tests_properties(validate_d3_profiles PROPERTIES
  PASS_REGULAR_EXPRESSION "V-14 PASS profiles=mizuno2025,mudirac130,parity pairs=[0-9]+\nV-15 PASS [^\n]*\nV-16 PASS profiles=mudirac130 keys=[0-9]+\n"
  FAIL_REGULAR_EXPRESSION " FAIL ")

# The assembled dataset with one byte appended to the level table's Layer-2 file: V-04 checks
# every loaded table's sibling, so it fails and names that table.
set(dir "${CMAKE_BINARY_DIR}/d3_digest_mismatch")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_DATASET_MEMBERS} DESTINATION "${dir}")
if(NOT EXISTS "${dir}/d3_levels.mudirac130.prov.json")
  message(FATAL_ERROR "the digest case needs d3_levels.mudirac130.prov.json in the assembled dataset")
endif()
file(APPEND "${dir}/d3_levels.mudirac130.prov.json" " ")
add_test(NAME validate_d3_digest_mismatch
         COMMAND g4muonicdata_validate "${dir}" --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_d3_digest_mismatch PROPERTIES
  PASS_REGULAR_EXPRESSION "V-04 FAIL [^\n]*level_energy")

# The level table without its first record: the key sets differ, and V-16 names the first key.
set(dir "${CMAKE_BINARY_DIR}/d3_key_sets_differ")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} "${G4MUONICDATA_D3_KSHELL}" DESTINATION "${dir}")
file(READ "${G4MUONICDATA_D3_LEVELS}" levels)
string(REGEX MATCH "\n[^#\n][^\n]*\n" first_record "${levels}")
if(first_record STREQUAL "")
  message(FATAL_ERROR "d3_levels.mudirac130.g4dat has no record line")
endif()
string(FIND "${levels}" "${first_record}" at)
string(LENGTH "${first_record}" length)
string(SUBSTRING "${levels}" 0 ${at} head)
math(EXPR rest_at "${at} + ${length} - 1")
string(SUBSTRING "${levels}" ${rest_at} -1 tail)
g4muonicdata_write_member("${dir}" "d3_levels.mudirac130.g4dat" "${head}${tail}")
add_test(NAME validate_d3_key_sets_differ
         COMMAND g4muonicdata_validate "${dir}" --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_d3_key_sets_differ PROPERTIES
  PASS_REGULAR_EXPRESSION "V-16 FAIL '#PROFILE mudirac130': k_shell_energy and level_energy differ first at key [0-9]+-[0-9]+ [(]carried by k_shell_energy only[)]\n")

# The k-shell table alone beside the D1 pair: V-16 names the missing table.
set(dir "${CMAKE_BINARY_DIR}/d3_one_table")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} "${G4MUONICDATA_D3_KSHELL}" DESTINATION "${dir}")
add_test(NAME validate_d3_one_table
         COMMAND g4muonicdata_validate "${dir}" --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_d3_one_table PROPERTIES
  PASS_REGULAR_EXPRESSION "V-16 FAIL '#PROFILE mudirac130' carries '#TABLE k_shell_energy' but not '#TABLE level_energy'\n")

# The k-shell table's A = 0 rows under A:listed: V-15 names the range.
set(dir "${CMAKE_BINARY_DIR}/d3_natural_under_listed")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_SHIPPED_FILES} "${G4MUONICDATA_D3_LEVELS}" DESTINATION "${dir}")
file(READ "${G4MUONICDATA_D3_KSHELL}" kshell)
string(FIND "${kshell}" " A:most_abundant_and_listed\n" at)
if(at EQUAL -1)
  message(FATAL_ERROR "d3_kshell.mudirac130.g4dat does not assign A:most_abundant_and_listed")
endif()
string(REPLACE " A:most_abundant_and_listed\n" " A:listed\n" kshell "${kshell}")
g4muonicdata_write_member("${dir}" "d3_kshell.mudirac130.g4dat" "${kshell}")
add_test(NAME validate_d3_natural_under_listed
         COMMAND g4muonicdata_validate "${dir}" --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
set_tests_properties(validate_d3_natural_under_listed PROPERTIES
  PASS_REGULAR_EXPRESSION "V-15 FAIL under A:listed: 'd3_kshell.mudirac130.g4dat' carries [0-9]+ A = 0 row")

# The Geant4-facing glue the patches add (cpp/src/G4MuonicDataOverlay.cc), compiled against the
# stand-in Geant4 headers under g4stub/ and run over the assembled dataset: each case checks what
# the glue answers against a resolution overlay_check makes itself, and any MISMATCH fails it.
add_executable(g4muonicdata_overlay_check overlay_check.cc ../src/G4MuonicDataOverlay.cc)
target_include_directories(g4muonicdata_overlay_check BEFORE PRIVATE g4stub ../include)
target_link_libraries(g4muonicdata_overlay_check PRIVATE g4muonicdatatable)
g4muonicdata_target_options(g4muonicdata_overlay_check)

# Every key and token below is read from the shipped files at configure time.
function(g4muonicdata_record_lines file out)
  file(STRINGS "${file}" lines REGEX "^ *[0-9]")
  if(NOT lines)
    message(FATAL_ERROR "${file} has no record line")
  endif()
  set(${out} "${lines}" PARENT_SCOPE)
endfunction()
g4muonicdata_record_lines("${G4MUONICDATA_D1_DIR}/d1_capture.g4dat" capture_lines)
list(GET capture_lines 0 line)
string(REGEX MATCH "^ *([0-9]+) +([0-9]+) " match "${line}")
set(capture_key "${CMAKE_MATCH_1} ${CMAKE_MATCH_2}")
g4muonicdata_record_lines("${G4MUONICDATA_D1_DIR}/d1_zeff.g4dat" zeff_lines)
list(GET zeff_lines 0 line)
string(REGEX MATCH "^ *([0-9]+) " match "${line}")
set(zeff_key "${CMAKE_MATCH_1}")
g4muonicdata_record_lines("${G4MUONICDATA_D3_KSHELL}" kshell_lines)
list(GET kshell_lines 0 line)
string(REGEX MATCH "^ *([0-9]+) +([0-9]+) " match "${line}")
set(kshell_key "${CMAKE_MATCH_1} ${CMAKE_MATCH_2}")
# The Z of every k-shell record, the first Z with a (Z, 0) record, and the lowest Z with none.
set(kshell_zs "")
set(natural_z "")
foreach(line IN LISTS kshell_lines)
  string(REGEX MATCH "^ *([0-9]+) +([0-9]+) " match "${line}")
  list(APPEND kshell_zs "${CMAKE_MATCH_1}")
  if(natural_z STREQUAL "" AND CMAKE_MATCH_2 STREQUAL "0")
    set(natural_z "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(natural_z STREQUAL "")
  message(FATAL_ERROR "d3_kshell.mudirac130.g4dat has no (Z, 0) record")
endif()
# A mass number equal to that Z, which no record of that Z carries, so it resolves to (Z, 0).
foreach(line IN LISTS kshell_lines)
  if(line MATCHES "^ *${natural_z} +${natural_z} ")
    message(FATAL_ERROR "d3_kshell.mudirac130.g4dat lists (${natural_z}, ${natural_z}); the unlisted key needs another A")
  endif()
endforeach()
set(unlisted_key "${natural_z} ${natural_z}")
set(missing_z 1)
list(FIND kshell_zs "${missing_z}" at)
while(NOT at EQUAL -1)
  math(EXPR missing_z "${missing_z} + 1")
  list(FIND kshell_zs "${missing_z}" at)
endwhile()
set(missing_key "${missing_z} ${missing_z}")
# The number of e<n> columns the level table declares, and the D3 profile token.
file(STRINGS "${G4MUONICDATA_D3_LEVELS}" level_columns REGEX "^#COLUMNS")
string(REGEX MATCHALL " e[0-9]+" level_e_columns "${level_columns}")
list(LENGTH level_e_columns level_count)
file(STRINGS "${G4MUONICDATA_D3_KSHELL}" d3_profile REGEX "^#PROFILE")
string(REGEX REPLACE "^#PROFILE +" "" d3_profile "${d3_profile}")
# The profiles the assembled dataset declares, bytewise sorted, as the glue lists them.
file(GLOB dataset_tables "${G4MUONICDATA_DATASET_DIR}/*.g4dat")
set(dataset_profiles "")
foreach(member IN LISTS dataset_tables)
  file(STRINGS "${member}" profile REGEX "^#PROFILE")
  string(REGEX REPLACE "^#PROFILE +" "" profile "${profile}")
  list(APPEND dataset_profiles "${profile}")
endforeach()
list(REMOVE_DUPLICATES dataset_profiles)
list(SORT dataset_profiles)
string(REPLACE ";" ", " dataset_profiles "${dataset_profiles}")
string(REPLACE " " ";" capture_args "rate ${capture_key}")
string(REPLACE " " ";" zeff_args "zeff ${zeff_key}")
string(REPLACE " " ";" kshell_args "kshell ${kshell_key}")
string(REPLACE " " ";" levels_args "levels ${kshell_key}")
string(REPLACE " " ";" unlisted_args "kshell ${unlisted_key}")
string(REPLACE " " ";" missing_args "kshell ${missing_key}")
string(REPLACE " " ";" missing_levels_args "levels ${missing_key}")

function(g4muonicdata_overlay_case name)
  cmake_parse_arguments(CASE "" "PASS" "ENV;QUERIES" ${ARGN})
  add_test(NAME ${name}
           COMMAND ${CMAKE_COMMAND} -E env ${CASE_ENV} $<TARGET_FILE:g4muonicdata_overlay_check> ${CASE_QUERIES})
  set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${CASE_PASS}" FAIL_REGULAR_EXPRESSION "MISMATCH")
endfunction()
set(parity_queries ${capture_args} ${zeff_args} ${kshell_args} ${levels_args})
set(parity_pass
    "CONFIG d1=parity d3=compiled dir=set version=[^\n]*\nTABLE loaded\nRATE ${capture_key} -?0x[^\n]* parity exact\nZEFF ${zeff_key} -?0x[^\n]* parity exact\nKSHELL ${kshell_key} none compiled compiled\nLEVELS ${kshell_key} none compiled compiled\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\n")

# The profile variable unset, then empty: parity, whose D1 keys hit and which carries no D3 table.
g4muonicdata_overlay_case(overlay_parity_default
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE
  QUERIES ${parity_queries} PASS "${parity_pass}")
g4muonicdata_overlay_case(overlay_profile_empty
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" "G4MUONICDATA_PROFILE="
  QUERIES ${parity_queries} PASS "${parity_pass}")

# The D3 profile named by the variable that spans both seams: no D1 table, so that seam is the
# compiled-in code; the first k-shell key hits exactly; an unlisted mass number of a Z that has
# a (Z, 0) record resolves nothing, because the profile is evaluated; the lowest Z with no
# record resolves nothing; the level lookup returns as many values as the level table declares
# e<n> columns.
g4muonicdata_overlay_case(overlay_${d3_profile}
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" "G4MUONICDATA_PROFILE=${d3_profile}"
  QUERIES ${capture_args} ${zeff_args} ${kshell_args} ${unlisted_args} ${missing_args} ${levels_args} ${missing_levels_args}
  PASS "CONFIG d1=compiled d3=${d3_profile} dir=set version=[^\n]*\nTABLE loaded\nRATE ${capture_key} none compiled compiled\nZEFF ${zeff_key} none compiled compiled\nKSHELL ${kshell_key} 0x[^\n]* ${d3_profile} exact\nKSHELL ${unlisted_key} none ${d3_profile} compiled\nKSHELL ${missing_key} none ${d3_profile} compiled\nLEVELS ${kshell_key} ${level_count} 0x[^\n]* ${d3_profile} exact\nLEVELS ${missing_key} none ${d3_profile} compiled\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\n")

# A profile no file declares: the fatal exception names the token and the profiles present; with
# the abort suppressed, every lookup resolves nothing and the run ends with every line a MATCH.
g4muonicdata_overlay_case(overlay_unknown_profile
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" "G4MUONICDATA_PROFILE=nosuchprofile"
  QUERIES ${kshell_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData004 [^\n]*'nosuchprofile'[^\n]*: ${dataset_profiles}\n")
g4muonicdata_overlay_case(overlay_suppressed_fatal
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" "G4MUONICDATA_PROFILE=nosuchprofile" "OVERLAY_CHECK_SUPPRESS_FATAL=1"
  QUERIES ${parity_queries}
  PASS "G4MuonicData004[^\n]*\nCONFIG d1=compiled d3=compiled dir=set version=[^\n]*\nTABLE loaded\nRATE ${capture_key} none compiled compiled\nZEFF ${zeff_key} none compiled compiled\nKSHELL ${kshell_key} none compiled compiled\nLEVELS ${kshell_key} none compiled compiled\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\n")

# No dataset directory resolves: the fatal exception the glue raises first.
g4muonicdata_overlay_case(overlay_no_dataset
  ENV --unset=G4MUONICDATA "G4MUONICDATA_PROFILE=${d3_profile}"
  QUERIES ${kshell_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData001 ")

# A k-shell table whose #UNITS assigns `value` another unit, and a level table without `e2` (its
# column renamed in #COLUMNS and #UNITS): the semantic layer refuses each before any table is
# selected, naming its code, the file, the table and what is wrong.
set(dir "${CMAKE_BINARY_DIR}/overlay_units_mismatch")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_DATASET_MEMBERS} DESTINATION "${dir}")
file(READ "${G4MUONICDATA_D3_KSHELL}" kshell)
string(FIND "${kshell}" " value=keV " at)
if(at EQUAL -1)
  message(FATAL_ERROR "d3_kshell.mudirac130.g4dat does not assign value=keV")
endif()
string(REPLACE " value=keV " " value=eV " kshell "${kshell}")
g4muonicdata_write_member("${dir}" "d3_kshell.mudirac130.g4dat" "${kshell}")
g4muonicdata_overlay_case(overlay_units_mismatch
  ENV "G4MUONICDATA=${dir}" "G4MUONICDATA_PROFILE=${d3_profile}"
  QUERIES ${kshell_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData005 [^\n]*S004 in file [^\n]*d3_kshell[^\n]*#TABLE k_shell_energy: #UNITS assigns 'value' the unit 'eV'")

set(dir "${CMAKE_BINARY_DIR}/overlay_no_e2")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_DATASET_MEMBERS} DESTINATION "${dir}")
file(READ "${G4MUONICDATA_D3_LEVELS}" levels)
foreach(old IN ITEMS " e2 " " e2=keV ")
  string(FIND "${levels}" "${old}" at)
  if(at EQUAL -1)
    message(FATAL_ERROR "d3_levels.mudirac130.g4dat does not carry '${old}'")
  endif()
endforeach()
string(REPLACE " e2 " " f2 " levels "${levels}")
string(REPLACE " e2=keV " " f2=keV " levels "${levels}")
g4muonicdata_write_member("${dir}" "d3_levels.mudirac130.g4dat" "${levels}")
g4muonicdata_overlay_case(overlay_no_e2
  ENV "G4MUONICDATA=${dir}" "G4MUONICDATA_PROFILE=${d3_profile}"
  QUERIES ${levels_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData005 [^\n]*S002 in file [^\n]*d3_levels[^\n]*#TABLE level_energy: #COLUMNS")

# The assembled dataset with every separating space of a multi-assignment #UNITS value replaced by
# one tab, which the format allows: the glue reads each assignment the same, for the default
# profile's D1 tables and for the D3 profile's tables.
set(dir "${CMAKE_BINARY_DIR}/overlay_units_tab")
file(REMOVE_RECURSE "${dir}")
file(COPY ${G4MUONICDATA_DATASET_MEMBERS} DESTINATION "${dir}")
set(tabbed_members "")
foreach(member IN LISTS G4MUONICDATA_DATASET_MEMBERS)
  if(NOT member MATCHES "[.]g4dat$")
    continue()
  endif()
  get_filename_component(name "${member}" NAME)
  file(READ "${member}" text)
  string(REGEX MATCH "\n#UNITS +([^\n]*)\n" match "${text}")
  if(match STREQUAL "")
    message(FATAL_ERROR "${name} carries no #UNITS line")
  endif()
  set(value "${CMAKE_MATCH_1}")
  if(value MATCHES " ")
    string(REPLACE " " "\t" tabbed_value "${value}")
    string(REPLACE "${value}" "${tabbed_value}" tabbed_match "${match}")
    string(REPLACE "${match}" "${tabbed_match}" text "${text}")
    g4muonicdata_write_member("${dir}" "${name}" "${text}")
    list(APPEND tabbed_members "${name}")
  endif()
endforeach()
foreach(name IN ITEMS d1_capture.g4dat d3_kshell.mudirac130.g4dat d3_levels.mudirac130.g4dat)
  if(NOT name IN_LIST tabbed_members)
    message(FATAL_ERROR "the tab case needs a multi-assignment #UNITS line in ${name}")
  endif()
endforeach()
g4muonicdata_overlay_case(overlay_units_tab_default
  ENV "G4MUONICDATA=${dir}" --unset=G4MUONICDATA_PROFILE
  QUERIES ${capture_args} ${zeff_args}
  PASS "RATE ${capture_key} -?0x[^\n]*\nZEFF ${zeff_key} -?0x[^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\n")
g4muonicdata_overlay_case(overlay_units_tab_${d3_profile}
  ENV "G4MUONICDATA=${dir}" "G4MUONICDATA_PROFILE=${d3_profile}"
  QUERIES ${kshell_args} ${levels_args}
  PASS "KSHELL ${kshell_key} 0x[^\n]*\nLEVELS ${kshell_key} ${level_count} 0x[^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\n")
set_tests_properties(overlay_units_tab_default overlay_units_tab_${d3_profile} PROPERTIES
  FAIL_REGULAR_EXPRESSION "MISMATCH|G4MuonicData003")

# The semantic layer (G4MuonicDataSemantics), which the validator reports as V-17 and the glue
# applies in production: the shipped dataset yields no issue, and one stated edit of one shipped
# member -- every replacement derived from that member's own bytes at configure time -- yields the
# code named here first. The mutation drill of each rule family lives in these cases.
add_test(NAME validate_semantics_shipped COMMAND g4muonicdata_validate ${G4MUONICDATA_VALIDATE_ARGS})
set_tests_properties(validate_semantics_shipped PROPERTIES
  PASS_REGULAR_EXPRESSION "V-17 PASS tables=[0-9]+ issues=0\n"
  FAIL_REGULAR_EXPRESSION " FAIL ")

function(g4muonicdata_semantics_case name member old new code)
  if(old STREQUAL "" OR new STREQUAL "" OR old STREQUAL new)
    message(FATAL_ERROR "${name}: the replacement is empty or changes nothing")
  endif()
  set(dir "${CMAKE_BINARY_DIR}/${name}")
  file(REMOVE_RECURSE "${dir}")
  file(COPY ${G4MUONICDATA_DATASET_MEMBERS} DESTINATION "${dir}")
  file(READ "${dir}/${member}" text)
  string(FIND "${text}" "${old}" at)
  if(at EQUAL -1)
    message(FATAL_ERROR "${name}: ${member} does not carry the text it mutates")
  endif()
  string(REPLACE "${old}" "${new}" mutated "${text}")
  g4muonicdata_write_member("${dir}" "${member}" "${mutated}")
  add_test(NAME ${name}
           COMMAND g4muonicdata_validate "${dir}" --oracle "${G4MUONICDATA_ORACLE}" --conformance "${G4MUONICDATA_CONFORMANCE_DIR}")
  set_tests_properties(${name} PROPERTIES
    PASS_REGULAR_EXPRESSION "V-17 FAIL [0-9]+ issue[(]s[)]; first: ${code} ")
endfunction()

get_filename_component(G4MUONICDATA_D3_KSHELL_NAME "${G4MUONICDATA_D3_KSHELL}" NAME)
# S001 -- the dataset name, prefixed in place.
file(STRINGS "${G4MUONICDATA_D3_KSHELL}" semantics_dataset_line REGEX "^#DATASET")
string(REGEX REPLACE "^(#DATASET +)(.+)$" "\\1other_\\2" semantics_dataset_other "${semantics_dataset_line}")
g4muonicdata_semantics_case(validate_semantics_dataset "${G4MUONICDATA_D3_KSHELL_NAME}"
  "${semantics_dataset_line}" "${semantics_dataset_other}" S001)

# S002 -- the D3 k-shell table declaring the D1 seam, copied from the D1 capture member's own line.
file(STRINGS "${G4MUONICDATA_D3_KSHELL}" semantics_seam_line REGEX "^#SEAM")
file(STRINGS "${G4MUONICDATA_D1_DIR}/d1_capture.g4dat" semantics_d1_seam_line REGEX "^#SEAM")
g4muonicdata_semantics_case(validate_semantics_seam "${G4MUONICDATA_D3_KSHELL_NAME}"
  "${semantics_seam_line}" "${semantics_d1_seam_line}" S002)

# S003 -- the declared Z range with its two bounds exchanged, so it is no longer inclusive.
file(STRINGS "${G4MUONICDATA_D3_KSHELL}" semantics_validity_line REGEX "^#VALIDITY")
string(REGEX REPLACE "Z:([0-9]+)-([0-9]+)" "Z:\\2-\\1" semantics_validity_swapped "${semantics_validity_line}")
g4muonicdata_semantics_case(validate_semantics_validity "${G4MUONICDATA_D3_KSHELL_NAME}"
  "${semantics_validity_line}" "${semantics_validity_swapped}" S003)

# S004 -- the unit of the column the K-energy lookup reads, prefixed so it is no longer the one the
# lookup reads it as.
file(STRINGS "${G4MUONICDATA_D3_KSHELL}" semantics_units_line REGEX "^#UNITS")
string(REGEX REPLACE "( value=)" "\\1x" semantics_units_prefixed "${semantics_units_line}")
g4muonicdata_semantics_case(validate_semantics_units "${G4MUONICDATA_D3_KSHELL_NAME}"
  "${semantics_units_line}" "${semantics_units_prefixed}" S004)

# S005 and S006 over the first k-shell record: its binding energy negated in place, and its binding
# energy replaced by its own uncertainty field, which is below the level the same key lists under
# `e2`. Both are built from that record's fields, so neither states a value.
list(GET kshell_lines 0 semantics_first_record)
string(REGEX MATCH "^( *[0-9]+ +[0-9]+ +) ([^ ]+)( +)([^ ]+)$" semantics_match "${semantics_first_record}")
if(semantics_match STREQUAL "")
  message(FATAL_ERROR "the semantic cases need the first k-shell record as `Z A value unc`")
endif()
set(semantics_head "${CMAKE_MATCH_1}")
set(semantics_value "${CMAKE_MATCH_2}")
set(semantics_gap "${CMAKE_MATCH_3}")
set(semantics_unc "${CMAKE_MATCH_4}")
g4muonicdata_semantics_case(validate_semantics_value "${G4MUONICDATA_D3_KSHELL_NAME}"
  "${semantics_first_record}" "${semantics_head}-${semantics_value}${semantics_gap}${semantics_unc}" S005)
g4muonicdata_semantics_case(validate_semantics_pair "${G4MUONICDATA_D3_KSHELL_NAME}"
  "${semantics_first_record}" "${semantics_head} ${semantics_unc}${semantics_gap}${semantics_unc}" S006)

# The two seams, selected independently; the exact-isotope rule; the frozen configuration; and the
# two computed-value guards. Every key below is read from the shipped files at configure time.
#
# From a two-key table's record lines: the first Z that carries both a (Z, 0) record and at least
# one (Z, A > 0) record, the lowest such A, and a mass number above every A that Z carries.
function(g4muonicdata_natural_probe file out_z out_listed out_absent)
  g4muonicdata_record_lines("${file}" lines)
  set(pairs "")
  foreach(line IN LISTS lines)
    string(REGEX MATCH "^ *([0-9]+) +([0-9]+) " match "${line}")
    if(match STREQUAL "")
      message(FATAL_ERROR "${file} has a record line that is not `Z A ...`")
    endif()
    list(APPEND pairs "${CMAKE_MATCH_1}:${CMAKE_MATCH_2}")
  endforeach()
  set(chosen "")
  set(listed "")
  set(highest 0)
  foreach(pair IN LISTS pairs)
    string(REPLACE ":" ";" parts "${pair}")
    list(GET parts 0 z)
    list(GET parts 1 a)
    if(NOT a EQUAL 0 OR NOT chosen STREQUAL "")
      continue()
    endif()
    foreach(other IN LISTS pairs)
      string(REPLACE ":" ";" fields "${other}")
      list(GET fields 0 oz)
      list(GET fields 1 oa)
      if(oz EQUAL z AND NOT oa EQUAL 0)
        set(chosen "${z}")
        if(listed STREQUAL "")
          set(listed "${oa}")
        endif()
        if(oa GREATER highest)
          set(highest "${oa}")
        endif()
      endif()
    endforeach()
  endforeach()
  if(chosen STREQUAL "")
    message(FATAL_ERROR "${file} carries no Z with both a natural-composition row and a listed isotope")
  endif()
  math(EXPR absent "${highest} + 1")
  set(${out_z} "${chosen}" PARENT_SCOPE)
  set(${out_listed} "${listed}" PARENT_SCOPE)
  set(${out_absent} "${absent}" PARENT_SCOPE)
endfunction()

set(G4MUONICDATA_D1_EVALUATED "${G4MUONICDATA_D1_DIR}/d1_capture.mizuno2025.g4dat")
if(NOT EXISTS "${G4MUONICDATA_D1_EVALUATED}")
  message(FATAL_ERROR "the seam cases need ${G4MUONICDATA_D1_EVALUATED}")
endif()
file(STRINGS "${G4MUONICDATA_D1_EVALUATED}" d1_profile REGEX "^#PROFILE")
string(REGEX REPLACE "^#PROFILE +" "" d1_profile "${d1_profile}")
g4muonicdata_natural_probe("${G4MUONICDATA_D1_EVALUATED}" evaluated_z evaluated_listed evaluated_absent)
string(REPLACE " " ";" evaluated_absent_args "rate ${evaluated_z} ${evaluated_absent}")
string(REPLACE " " ";" evaluated_natural_args "rate ${evaluated_z} 0")
string(REPLACE " " ";" evaluated_listed_args "rate ${evaluated_z} ${evaluated_listed}")

# Each variable alone, both together, the one that spans both seams, and none: what each selects.
g4muonicdata_overlay_case(overlay_seam_d1_alone
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE --unset=G4MUONICDATA_D3_PROFILE
      "G4MUONICDATA_D1_PROFILE=${d1_profile}"
  QUERIES ${evaluated_listed_args}
  PASS "CONFIG d1=${d1_profile} d3=compiled dir=set version=[^\n]*\nTABLE loaded\nRATE ${evaluated_z} ${evaluated_listed} 0x[^\n]* ${d1_profile} exact\nMATCH [^\n]*\n")
g4muonicdata_overlay_case(overlay_seam_d3_alone
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE --unset=G4MUONICDATA_D1_PROFILE
      "G4MUONICDATA_D3_PROFILE=${d3_profile}"
  QUERIES ${capture_args} ${kshell_args}
  PASS "CONFIG d1=parity d3=${d3_profile} dir=set version=[^\n]*\nTABLE loaded\nRATE ${capture_key} 0x[^\n]* parity exact\nKSHELL ${kshell_key} 0x[^\n]* ${d3_profile} exact\nMATCH [^\n]*\nMATCH [^\n]*\n")
g4muonicdata_overlay_case(overlay_seams_together
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE
      "G4MUONICDATA_D1_PROFILE=${d1_profile}" "G4MUONICDATA_D3_PROFILE=${d3_profile}"
  QUERIES ${evaluated_listed_args} ${kshell_args}
  PASS "CONFIG d1=${d1_profile} d3=${d3_profile} dir=set version=[^\n]*\nTABLE loaded\nRATE ${evaluated_z} ${evaluated_listed} 0x[^\n]* ${d1_profile} exact\nKSHELL ${kshell_key} 0x[^\n]* ${d3_profile} exact\nMATCH [^\n]*\nMATCH [^\n]*\n")
# A per-seam variable wins over the one that spans both seams; an empty variable is unset.
g4muonicdata_overlay_case(overlay_seam_over_legacy
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" "G4MUONICDATA_PROFILE=${d3_profile}"
      "G4MUONICDATA_D1_PROFILE=${d1_profile}"
  QUERIES ${evaluated_listed_args} ${kshell_args}
  PASS "CONFIG d1=${d1_profile} d3=${d3_profile} dir=set version=[^\n]*\n")
g4muonicdata_overlay_case(overlay_seam_empty_is_unset
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" "G4MUONICDATA_PROFILE=${d3_profile}" "G4MUONICDATA_D1_PROFILE="
      "G4MUONICDATA_D3_PROFILE="
  QUERIES ${kshell_args}
  PASS "CONFIG d1=compiled d3=${d3_profile} dir=set version=[^\n]*\n")

# The exact-isotope rule under an evaluated profile: the listed isotope and the natural-composition
# row each answer a request for themselves, and a mass number the profile does not list resolves
# nothing -- where `parity` would fall through to the (Z, 0) record.
g4muonicdata_overlay_case(overlay_exact_isotope
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE --unset=G4MUONICDATA_D3_PROFILE
      "G4MUONICDATA_D1_PROFILE=${d1_profile}"
  QUERIES ${evaluated_absent_args} ${evaluated_natural_args} ${evaluated_listed_args}
  PASS "RATE ${evaluated_z} ${evaluated_absent} none ${d1_profile} compiled\nRATE ${evaluated_z} 0 0x[^\n]* ${d1_profile} explicit_natural\nRATE ${evaluated_z} ${evaluated_listed} 0x[^\n]* ${d1_profile} exact\nMATCH [^\n]*\nMATCH [^\n]*\nMATCH [^\n]*\n")

# `compiled` on one seam, and on both: with both seams compiled no directory is searched and
# nothing is read, which is shown by naming a directory that does not exist.
g4muonicdata_overlay_case(overlay_compiled_one_seam
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE "G4MUONICDATA_D3_PROFILE=compiled"
  QUERIES ${capture_args} ${kshell_args}
  PASS "CONFIG d1=parity d3=compiled dir=set version=[^\n]*\nTABLE loaded\nRATE ${capture_key} 0x[^\n]* parity exact\nKSHELL ${kshell_key} none compiled compiled\nMATCH [^\n]*\nMATCH [^\n]*\n")
g4muonicdata_overlay_case(overlay_compiled_both_seams
  ENV "G4MUONICDATA=${CMAKE_BINARY_DIR}/no_such_dataset_directory" "G4MUONICDATA_PROFILE=compiled"
  QUERIES ${capture_args} ${kshell_args}
  PASS "CONFIG d1=compiled d3=compiled dir=none version=none\nTABLE none\nRATE ${capture_key} none compiled compiled\nKSHELL ${kshell_key} none compiled compiled\nMATCH [^\n]*\nMATCH [^\n]*\n")
set_tests_properties(overlay_compiled_both_seams PROPERTIES FAIL_REGULAR_EXPRESSION "MISMATCH|G4Exception")

# A seam named explicitly whose profile carries no table of that seam: fatal, naming the variable.
g4muonicdata_overlay_case(overlay_seam_d1_missing
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE "G4MUONICDATA_D1_PROFILE=${d3_profile}"
  QUERIES ${capture_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData004 G4MUONICDATA_D1_PROFILE names '${d3_profile}'[^\n]*nuclear_capture_rate")
g4muonicdata_overlay_case(overlay_seam_d3_missing
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE "G4MUONICDATA_D3_PROFILE=${d1_profile}"
  QUERIES ${kshell_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData004 G4MUONICDATA_D3_PROFILE names '${d1_profile}'[^\n]*k_shell_energy")

# The opt-in clear: no lookup, no configuration and no I/O, over a directory that cannot be read
# and over one whose member the reader refuses.
g4muonicdata_overlay_case(overlay_off_unreadable
  ENV "G4MUONICDATA=${CMAKE_BINARY_DIR}/no_such_dataset_directory" --unset=G4MUONICDATA_PROFILE
  QUERIES --off ${capture_args} ${kshell_args}
  PASS "CONFIG d1=compiled d3=compiled dir=none version=none\nTABLE none\nRATE ${capture_key} none compiled compiled\nKSHELL ${kshell_key} none compiled compiled\nMATCH [^\n]*\nMATCH [^\n]*\n")
g4muonicdata_overlay_case(overlay_off_malformed
  ENV "G4MUONICDATA=${CMAKE_BINARY_DIR}/discovery_invalid" --unset=G4MUONICDATA_PROFILE
  QUERIES --off ${capture_args}
  PASS "CONFIG d1=compiled d3=compiled dir=none version=none\nTABLE none\nRATE ${capture_key} none compiled compiled\nMATCH [^\n]*\n")
set_tests_properties(overlay_off_unreadable overlay_off_malformed PROPERTIES
  FAIL_REGULAR_EXPRESSION "MISMATCH|G4Exception")

# The environment is read once: a profile exported after the configuration was frozen selects
# nothing, and the answers stay the ones the environment at start-up selected.
g4muonicdata_overlay_case(overlay_frozen_environment
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE
  QUERIES --init --setenv "G4MUONICDATA_PROFILE=${d3_profile}" ${capture_args} ${kshell_args}
  PASS "CONFIG d1=parity d3=compiled dir=set version=[^\n]*\nTABLE loaded\nRATE ${capture_key} 0x[^\n]* parity exact\nKSHELL ${kshell_key} none compiled compiled\nMATCH [^\n]*\nMATCH [^\n]*\n")

# A dataset the semantic layer refuses is refused in production too: fatal, naming the code, the
# file and the table; and with the abort suppressed the glue answers nothing at all rather than
# from a dataset no consumer may read.
g4muonicdata_overlay_case(overlay_semantics_fatal
  ENV "G4MUONICDATA=${CMAKE_BINARY_DIR}/validate_semantics_value" "G4MUONICDATA_PROFILE=${d3_profile}"
  QUERIES ${kshell_args}
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData005 [^\n]*S005 in file [^\n]*d3_kshell[^\n]*k_shell_energy")
g4muonicdata_overlay_case(overlay_semantics_suppressed
  ENV "G4MUONICDATA=${CMAKE_BINARY_DIR}/validate_semantics_pair" "G4MUONICDATA_PROFILE=${d3_profile}"
      "OVERLAY_CHECK_SUPPRESS_FATAL=1"
  QUERIES ${kshell_args}
  PASS "G4MuonicData005 [^\n]*S006[^\n]*\nCONFIG [^\n]*\nREFUSED [^\n]*S006")

# The computed capture rate: under an evaluated profile a value that is not a positive finite rate
# is fatal, and under `parity` the compiled-in model's documented output stands.
g4muonicdata_overlay_case(overlay_computed_rate_evaluated
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE "G4MUONICDATA_D1_PROFILE=${d1_profile}"
  QUERIES checkrate ${evaluated_z} ${evaluated_absent} negative
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData005 under #PROFILE ${d1_profile}[^\n]*not a positive finite rate")
g4muonicdata_overlay_case(overlay_computed_rate_nonfinite
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE "G4MUONICDATA_D1_PROFILE=${d1_profile}"
  QUERIES checkrate ${evaluated_z} ${evaluated_absent} nan
  PASS "G4Exception G4MuonicDataOverlay G4MuonicData005 [^\n]*not a positive finite rate")
g4muonicdata_overlay_case(overlay_computed_rate_parity
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE
  QUERIES checkrate ${evaluated_z} ${evaluated_absent} negative checkrate ${evaluated_z} ${evaluated_absent} nan
  PASS "CHECKRATE ${evaluated_z} ${evaluated_absent} negative returned\nCHECKRATE ${evaluated_z} ${evaluated_absent} nan returned\n")
set_tests_properties(overlay_computed_rate_parity PROPERTIES FAIL_REGULAR_EXPRESSION "MISMATCH|G4Exception")

# The assembled muonic-cascade array: a falling one passes, and one that is flat, negative or the
# wrong length is fatal.
g4muonicdata_overlay_case(overlay_cascade_levels_falling
  ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE
  QUERIES checklevels falling
  PASS "CHECKLEVELS falling returned\n")
set_tests_properties(overlay_cascade_levels_falling PROPERTIES FAIL_REGULAR_EXPRESSION "MISMATCH|G4Exception")
foreach(shape IN ITEMS flat negative short)
  g4muonicdata_overlay_case(overlay_cascade_levels_${shape}
    ENV "G4MUONICDATA=${G4MUONICDATA_DATASET_DIR}" --unset=G4MUONICDATA_PROFILE
    QUERIES checklevels ${shape}
    PASS "G4Exception G4MuonicDataOverlay G4MuonicData005 the muonic cascade assembled ")
endforeach()

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
