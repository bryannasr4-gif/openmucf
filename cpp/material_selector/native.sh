#!/usr/bin/env bash
# Build the harness against one Geant4 installation and run one named batch of cases. Every command, its exit
# code and the identities of the sources, headers, executable and linked libraries are written under --work.
#
#   native.sh build   <tag> <install> <work> [drills] [asan]
#   native.sh run     <tag> <work> <catalogue> <physics-data> <batch>
#
# Batches: probe, bind, order, pilot (the aluminium ladder matrix), pbo, mutants, drill, identity,
# identity_data, teardown.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd="$1"; shift

log() { printf '%s\n' "$*" >> "$work/commands.log"; }

run_case() {  # name, then the harness arguments
  local name="$1"; shift
  local out="$work/runs/$name"
  mkdir -p "$out"
  local start rc
  start=$(date +%s.%N)
  set +e
  env -i PATH=/usr/bin:/bin HOME="$work" LD_LIBRARY_PATH="$install/lib" GEANT4_DATA_DIR="$data" \
    ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" "$binary" --out "$out" "$@" > "$out/stdout.log" 2>&1
  rc=$?
  set -e
  log "CASE $name rc=$rc wall_s=$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}') args=$*"
  echo "$rc" > "$out/rc"
}

if [ "$cmd" = build ]; then
  tag="$1"; install="$2"; work="$3"; drills="${4:-}"; asan="${5:-OFF}"
  mkdir -p "$work"
  : > "$work/commands.log"
  geant4_dir="$(dirname "$(find "$install" -name Geant4Config.cmake | head -1)")"
  cmake -S "$here" -B "$work/build" -DCMAKE_BUILD_TYPE=Release -DGeant4_DIR="$geant4_dir" \
    -DG4MUONIC_DRILLS="$drills" -DG4MUONIC_ASAN="$asan" > "$work/configure.log" 2>&1
  cmake --build "$work/build" -j 4 > "$work/compile.log" 2>&1
  binary="$work/build/harvest_material"
  {
    echo "tag $tag"
    echo "install $install"
    echo "geant4_config_version $("$install/bin/geant4-config" --version)"
    echo "multithreading $("$install/bin/geant4-config" --has-feature multithreading)"
    echo "gxx $(g++ --version | head -1)"
    cat "$work/build/identity.txt"
    for f in "$here"/G4MuonicMaterialSelector.hh "$here"/G4MuonicMaterialSelector.cc "$here"/harvest_material.cc \
             "$here"/CMakeLists.txt "$here"/../src/G4MuonicDataTable.cc "$here"/../include/G4MuonicDataTable.hh; do
      echo "source $(basename "$f") $(sha256sum "$f" | cut -d' ' -f1)"
    done
    for h in G4MuonMinusAtomicCapture.hh G4HadronStoppingProcess.hh G4ElementSelector.hh; do
      echo "header $h $(sha256sum "$(find "$install/include" -name "$h" | head -1)" | cut -d' ' -f1)"
    done
    echo "executable $(sha256sum "$binary" | cut -d' ' -f1)"
    LD_LIBRARY_PATH="$install/lib" ldd "$binary" | awk '/=> \//{print $3}' | sort | while read -r lib; do
      echo "library $(basename "$lib") $(sha256sum "$lib" | cut -d' ' -f1)"
    done
  } > "$work/build.txt"
  echo "BUILT $tag $(grep '^executable' "$work/build.txt")"
  exit 0
fi

tag="$1"; work="$2"; catalogue="$3"; data="$4"; batch="$5"
binary="$work/build/harvest_material"
install="$(awk '$1=="install"{print $2}' "$work/build.txt")"
N=1048576
case "$batch" in
  probe)
    run_case probe --mode probe --catalogue "$catalogue" --draws "$N" ;;
  bind)
    run_case bind --mode bind --catalogue "$catalogue" ;;
  order)
    run_case order_after --mode order --catalogue "$catalogue" --order after
    run_case order_before --mode order --catalogue "$catalogue" --order before ;;
  pilot)
    for route in stopping atomic; do
      for manager in serial tasking; do
        for enable in absent off on off_bound on_unbound; do
          run_case "ladder_${route}_${manager}_${enable}" --mode ladder --route "$route" --manager "$manager" \
            --threads 2 --material al2o3 --enable "$enable" --catalogue "$catalogue" --draws "$N"
        done
        run_case "ladder_${route}_${manager}_on_rev" --mode ladder --route "$route" --manager "$manager" \
          --threads 2 --material al2o3_rev --enable on --catalogue "$catalogue" --draws "$N"
      done
    done ;;
  pbo)
    for route in stopping atomic; do
      for manager in serial tasking; do
        for material in pbo_nist pbo_rev; do
          run_case "ladder_${route}_${manager}_${material}" --mode ladder --route "$route" --manager "$manager" \
            --threads 2 --material "$material" --enable on --catalogue "$catalogue" --draws "$N"
        done
      done
    done ;;
  mutants)  # one route and manager per mutant table; the case names carry the catalogue (the mutant)
    for route in stopping atomic; do
      run_case "mut_$(basename "$catalogue")_ladder_${route}_serial_on" --mode ladder --route "$route" --manager serial --material al2o3 \
        --enable on --catalogue "$catalogue" --draws "$N"
      run_case "mut_$(basename "$catalogue")_ladder_${route}_serial_off_bound" --mode ladder --route "$route" --manager serial --material al2o3 \
        --enable off_bound --catalogue "$catalogue" --draws "$N"
    done ;;
  drill)  # the cases the mutation-drill builds are read on
    for c in "stopping serial al2o3 on" "stopping serial al2o3 off_bound" "stopping tasking al2o3 on" \
             "stopping serial al2o3_rev on"; do
      set -- $c
      suffix="$4"; [ "$3" = al2o3_rev ] && suffix=on_rev
      run_case "ladder_$1_$2_${suffix}" --mode ladder --route "$1" --manager "$2" --threads 2 --material "$3" \
        --enable "$4" --catalogue "$catalogue" --draws "$N"
    done ;;
  noio)  # disabled, the provider reads nothing: a missing or malformed catalogue path changes nothing
    mkdir -p "$work/malformed" && printf '#GRAMMAR 9.0\n' > "$work/malformed/bad.g4dat"
    for path in /nonexistent-catalogue "$work/malformed"; do
      tagp=$([ "$path" = /nonexistent-catalogue ] && echo missing || echo malformed)
      run_case "noio_${tagp}_off" --mode identity --route stopping --manager serial --material al2o3 \
        --enable off --catalogue "$path" --events 100 --seed-base 17
      run_case "noio_${tagp}_on" --mode identity --route stopping --manager serial --material al2o3 \
        --enable on --catalogue "$path" --events 100 --seed-base 17
    done ;;
  identity)
    for route in stopping atomic; do
      for manager in serial tasking; do
        for seed in 17 101 1009; do
          for enable in absent off off_bound on_unbound; do
            run_case "identity_${route}_${manager}_${seed}_${enable}" --mode identity --route "$route" \
              --manager "$manager" --threads 2 --material al2o3 --enable "$enable" --catalogue "$catalogue" \
              --events 100 --seed-base "$seed"
          done
        done
      done
    done ;;
  identity_data)
    for route in stopping atomic; do
      for manager in serial tasking; do
        for seed in 17 101 1009; do
          for enable in absent off; do
            run_case "identity_data_${route}_${manager}_${seed}_${enable}" --mode identity --route "$route" \
              --manager "$manager" --threads 2 --material al2o3 --enable "$enable" --data on \
              --catalogue "$catalogue" --events 100 --seed-base "$seed"
          done
        done
      done
      run_case "ladder_data_${route}_serial_on" --mode ladder --route "$route" --manager serial --material al2o3 \
        --enable on --data on --catalogue "$catalogue" --draws "$N"
    done ;;
  teardown)
    for route in stopping atomic; do
      run_case "teardown_${route}" --mode teardown --route "$route" --manager tasking --threads 2 \
        --material al2o3 --enable on --catalogue "$catalogue" --events 50
    done ;;
  *) echo "unknown batch $batch" >&2; exit 2 ;;
esac
echo "RAN $tag $batch $(grep -c '^CASE' "$work/commands.log") cases so far; nonzero: $(grep -c 'rc=[1-9]' "$work/commands.log" || true)"
