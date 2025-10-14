#!/bin/bash

# Ensure script exits on any error
set -e

# --- Configuration and Initialization ---

# Initialize conda
echo "Initializing conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
CONDA_ENV_NAME="iccad"
if ! conda activate "$CONDA_ENV_NAME"; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
    exit 1
fi
echo "Environment '$CONDA_ENV_NAME' activated."

# Set environment variables
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/:$LD_LIBRARY_PATH"
# Setting RUSTFLAGS for general optimization and rpath for linked libraries
export RUSTFLAGS="-C link-arg=-Wl,-O2,-rpath,${CONDA_PREFIX}/lib -C target-cpu=native"
export RUST_BACKTRACE=1

# --- Argument Parsing and Execution ---

# Function to display usage information
function show_usage {
    echo "Usage: $0 <mode>"
    echo ""
    echo "Available modes:"
    echo "  debug      : Standard debug build and run (cargo run)"
    echo "  release    : Standard release build and run (cargo run --release)"
    echo "  pgo        : Profile-Guided Optimization build and run"
    echo "  profile    : Profiling with 'perf' (requires sudo/permissions to run perf)"
    echo "  flame      : Generate a CPU flamegraph"
    echo "  hotpath    : Run with the 'hotpath' feature enabled"
    echo ""
    echo "Example: $0 release"
}

# Check if an argument was provided
if [ -z "$1" ]; then
    show_usage
    exit 1
fi

MODE="$1"

echo "----------------------------------------"
echo "Running mbffg in mode: $MODE"
echo "----------------------------------------"

case "$MODE" in
    debug)
        echo "--> Executing standard debug run."
        cargo run
        ;;

    release)
        echo "--> Executing standard release run."
        cargo run --release
        ;;

    pgo)
        echo "--> Executing Profile-Guided Optimization (PGO) sequence."
        # Note: 'cargo install cargo-pgo' should be run once if not already installed
        cargo pgo instrument run
        cargo pgo optimize build
        echo "--> Running optimized PGO binary..."
        ./target/release/mbffg
        ;;

    profile)
        echo "--> Executing release build and profiling with 'perf'."
        # RUSTFLAGS are configured above, assuming they are sufficient for symbols
        cargo build --release
        echo "--> Starting perf record..."
        # NOTE: You may need to run this command with 'sudo' depending on your system settings.
        perf record -F 999 -e cycles:u -- ./target/release/mbffg
        echo "--> Generating perf report. Use 'perf report' manually for interactive viewing."
        perf report
        ;;

    flame)
        echo "--> Executing release build and generating flamegraph."
        # Note: 'cargo install cargo-flamegraph' should be run once if not already installed
        cargo flamegraph --release
        ;;

    hotpath)
        echo "--> Executing run with 'hotpath' feature."
        cargo run --features hotpath
        ;;

    *)
        echo "Error: Unknown mode '$MODE'."
        show_usage
        exit 1
        ;;
esac

echo "----------------------------------------"
echo "Execution finished for mode: $MODE"
echo "----------------------------------------"
