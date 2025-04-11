#!/bin/bash

# Ensure script exits on any error
set -e

# Set environment variables
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH"
export RUST_LOG="debug,main=error,geo=info"
export RUSTFLAGS="-C overflow-checks=yes -Awarnings"
export RUST_BACKTRACE=1

# Run the Rust binary
cargo run --release \
  --bin hello_world
  # --manifest-path hello_world/Cargo.toml \
