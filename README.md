# Parallel Utility-Guided Multi-Bit Flip-Flop Clustering

Multibit-Flip-Flop Generator and Evaluation Toolkit.

This repository serves as the open-source implementation accompanying the research paper *"Parallel Utility-Guided Multi-Bit Flip-Flop Clustering with Timing and Power Co-Optimization"* on multibit flip-flop grouping (MBFFG).

This repository implements the algorithmic framework proposed in the paper for compiling, generating, and evaluating multibit flip-flop groupings (MBFFG). It includes scripts for building optimized binaries, performing sanity checks, and reproducing experimental results in the research paper. The benchmark dataset used in this work is derived from **Problem B** of the ICCAD 2024 contest (see [https://www.iccad-contest.org/2024/Problems.html](https://www.iccad-contest.org/2024/Problems.html)).

## Building the Project

### Compile with PGO

To build the project using profiling-guided optimization, run:

```zsh
cd Mbffg && zsh run.sh pgo
```

This script orchestrates compilation, profiling, and final PGO-optimized production of the MBFFG binary.

## Precompiled Binary

After compilation, the optimized binary is located at:

```zsh
Mbffg/target/x86_64-unknown-linux-gnu/release/mbffg
```

You may invoke it directly for your experiments or integration workflows.

## Usage

There are two primary ways to run the MBFFG binary:

1. **Run all testcases**

   Executes all available contest testcases sequentially:

   ```zsh
   cd Mbffg && target/x86_64-unknown-linux-gnu/release/mbffg
   ```

2. **Run a specific testcase (c1–c7)**

   Executes only the specified testcase (e.g., `c1`, `c2`, ..., `c7`):

   ```zsh
   cd Mbffg && target/x86_64-unknown-linux-gnu/release/mbffg c3
   ```

   Replace `c3` with any testcase ID from `c1` to `c7` to run the corresponding testcase.

## Reproducing Paper Results

### Timing Evaluation Script

The script used for benchmarking in the paper is available at:

```zsh
experimental_results/benchmark.sh
```

Run it to reproduce performance evaluations:

```zsh
cd Mbffg && zsh ../experimental_results/benchmark.sh
```

### Sanity Check

Content-organizer sanity checks used in the paper can be run with:

```zsh
cd Mbffg && zsh ../experimental_results/sanity_check.sh
```

## Experimental Outputs

All experimental outputs referenced in the paper are stored in:

```zsh
experimental_results/*.out
```

## Citation

If you use this software in academic work, please cite the accompanying paper.
