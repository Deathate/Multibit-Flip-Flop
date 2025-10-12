#!/bin/bash

# Ensure script exits on any error
set -e

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate iccad

# Set environment variables
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/:$LD_LIBRARY_PATH"
export RUSTFLAGS="-C overflow-checks=yes -C link-arg=-Wl,-rpath,${CONDA_PREFIX}/lib "
export RUST_LOG="info"
export RUST_BACKTRACE=1

cargo run --release \
  # --no-default-features
  # --manifest-path hello_world/Cargo.toml \

# RUSTFLAGS="-C overflow-checks=yes -C link-arg=-Wl,-rpath,${CONDA_PREFIX}/lib -C target-cpu=native -C link-arg=--emit-relocs -C force-frame-pointers=yes" \
# cargo build --release
