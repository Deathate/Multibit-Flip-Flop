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
    bool output);