#pragma once
#include "hello_world/src/bridge_cxx.rs.h"
#include "rust/cxx.h"
// void test();
// int add(int a, int b);
// void print_message_from_rust(rust::Vec<NodeInfo> elements);
rust::Vec<int> solveTilingProblem(
    const Tuple2_int &gridSize,
    const rust::Vec<Tuple2_int> &tiles_arg,
    const rust::Vec<double> &tileWeights,
    const rust::Vec<int> &tileLimits,
    const rust::Vec<List_int> &spatialOccupancy,
    bool output);