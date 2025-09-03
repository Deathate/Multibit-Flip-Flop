#!/bin/bash

# Ensure script exits on any error
set -e

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate iccad

# Set environment variables
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/:$LD_LIBRARY_PATH"
export RUST_LOG="debug,hello_world=debug,geo=info"
export RUSTFLAGS="-C overflow-checks=yes -Awarnings"
export RUST_BACKTRACE=1

# --- Gurobi setup ---
export GUROBI_HOME="/opt/gurobi/gurobi1201/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${LIBRARY_PATH}:${GUROBI_HOME}/lib"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${GUROBI_HOME}/include"

# --- crate tch setup ---
export LIBTORCH="${HOME}/libtorch"
# LIBTORCH_INCLUDE must contain `include` directory.
export LIBTORCH_INCLUDE="${LIBTORCH}/"
# LIBTORCH_LIB must contain `lib` directory.
export LIBTORCH_LIB="${LIBTORCH}/"
export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH}"

# Run the Rust binary
cargo run --release \
  --bin hello_world \
  --no-default-features
  # --manifest-path hello_world/Cargo.toml \
