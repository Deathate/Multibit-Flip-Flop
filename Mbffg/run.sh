#!/bin/bash

# Ensure script exits on any error
set -e

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate iccad

# Set environment variables
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/:$LD_LIBRARY_PATH"
export RUSTFLAGS="-C link-arg=-Wl,-O2,-rpath,${CONDA_PREFIX}/lib -C target-cpu=native "
export RUST_BACKTRACE=1
export RAYON_NUM_THREADS=24

RUST_LOG=debug cargo run

# cargo clean
# cargo run --release

# cargo install cargo-pgo
# cargo pgo instrument run
# cargo pgo optimize build
# RUST_LOG="debug" target/release/mbffg

# cargo build --release
# RUSTFLAGS="-C link-arg=--emit-relocs -C force-frame-pointers=yes" \
#   cargo build --release

# export RUSTFLAGS="$RUSTFLAGS -C debuginfo=2 -C link-arg=-Wl,--emit-relocs"
# cargo build --release
# sudo sysctl -w kernel.perf_event_paranoid=-1
# sudo sysctl -w kernel.kptr_restrict=0
# sudo perf record -F 999 -g -- ./target/release/mbffg
# # sudo perf record -e cycles:u -c 1000 -j any,u -o perf.data -- ./target/release/mbffg
# sudo chown deathate:deathate perf.data
# hotspot
# perf report

# CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --release

# cargo run --features hotpath

# sudo perf record -e cycles:u -c 1000 -j any,u -o perf.data -- ./target/release/mbffg
# sudo perf2bolt ./target/release/mbffg -p perf.data -o perf.fdata
# llvm-bolt ./target/release/mbffg -o ./target/release/mbffg.bolt -data=perf.fdata -reorder-blocks=ext-tsp -reorder-functions=cdsort -jump-tables=aggressive -split-functions -split-all-cold
# ./target/release/mbffg
# ./target/release/mbffg.bolt
