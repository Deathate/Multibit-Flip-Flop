#pragma once

#include "hello_world/src/bridge_cxx.rs.h"
#include "rust/cxx.h"
// void test();
// int add(int a, int b);
// void print_message_from_rust(rust::Vec<NodeInfo> elements);
rust::Vec<SpatialInfo> solveTilingProblem(
    const Tuple2_int gridSize,
    const rust::Vec<TileInfo> tileInfos,
    const rust::Vec<List_int> spatialOccupancy,
    const int split,
    bool output);
rust::Vec<List_int> solveMultipleKnapsackProblem(
    const rust::Vec<Pair_Int_ListFloat> items,
    const rust::Vec<int> knapsack_capacities);
// rust::Vec<List_bool> solve_tiling_problem(
//     const rust::Vec<List_bool> cover_map,
//     const Tuple2_int tile_size);
