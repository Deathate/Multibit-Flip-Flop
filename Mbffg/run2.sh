#!/usr/bin/env bash
set -euo pipefail

# --- Conda (as you had) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate iccad

# --- Dynamic rpath for the conda env ---
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --- LLVM utils from the Rust toolchain (matches your rustc) ---
rustup component add llvm-tools-preview >/dev/null 2>&1 || true
LLVM_BIN="$(rustc --print target-libdir)/../bin"
"${LLVM_BIN}/llvm-profdata" --version

# --- 1) Clean PGO dir & build instrumented binary ---
rm -rf pgo
mkdir -p pgo

# Append (don’t overwrite) the instrument flag for this *one* build
RUSTFLAGS="-Cprofile-generate=./pgo" \
  cargo build --release

# --- 2) Run representative workloads (repeat as needed) ---
# (No trailing backslashes before comments!)
./target/release/mbffg

# --- 3) Merge raw profiles into a single .profdata ---
# Ensure at least one .profraw exists
shopt -s nullglob
profraws=(pgo/*.profraw)
if (( ${#profraws[@]} == 0 )); then
  echo "No .profraw files found in ./pgo. Did the instrumented binary run?"
  exit 1
fi
"${LLVM_BIN}/llvm-profdata" merge -o ./pgo/merged.profdata "${profraws[@]}"

# --- 4) Final optimized build with PGO + Fat LTO ---
# Append PGO/LTO flags without losing earlier ones (like rpath)
RUSTFLAGS="-Cprofile-use=./pgo/merged.profdata -Cllvm-args=--pgo-warn-missing-function -Clto=fat" \
  cargo clean && cargo build --release

# RUST_LOG="info" ./target/release/mbffg
RUSTFLAGS="-C link-arg=--emit-relocs -C force-frame-pointers=yes" \
  cargo build --release

sudo perf record -e cycles:u -j any,u -o perf.data -- ./target/release/mbffg
sudo perf2bolt ./target/release/mbffg -p perf.data -o perf.fdata
llvm-bolt ./target/release/mbffg -o ./target/release/mbffg.bolt \
  -data=perf.fdata -reorder-blocks=ext-tsp -reorder-functions=cdsort \
  -jump-tables=aggressive -split-functions -split-all-cold
RUST_LOG="info" ./target/release/mbffg
RUST_LOG="info" ./target/release/mbffg.bolt
