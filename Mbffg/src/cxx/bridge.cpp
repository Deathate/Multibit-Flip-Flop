#include <iostream>
#include <ranges>
// #define NDEBUG
#include <cassert>
#include <thread>
#ifdef __APPLE__
#include "/Library/gurobi1200/macos_universal2/include/gurobi_c++.h"
#elif __linux__
#include "gurobi_c++.h"
#else
std::cout << "Unknown operating system" << std::endl;
#endif
#include "bridge.h"
#include "formatter.cpp"
#include "print.hpp"
using namespace std;

GRBEnv env;

void start_env() {
    bool env_start = false;
    if (!env_start) {
        // env.set(GRB_IntParam_LogToConsole, 0);
        env.start();
        env_start = true;
    }
}

void print_message_from_rust() {
    try {
        // Create environment
        start_env();

        // Create a model
        GRBModel model = GRBModel(env);

        // Create variables
        GRBVar x = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x");
        GRBVar y = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "y");
        GRBVar z = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "z");

        // Set objective: Maximize 3x + y + 2z
        model.setObjective(3 * x + y + 2 * z, GRB_MAXIMIZE);

        // Add constraints
        model.addConstr(x + 2 * y + 3 * z <= 4, "c0");
        model.addConstr(x + y >= 1, "c1");

        // Optimize the model
        model.optimize();

        // Display the results
        cout << "Optimal objective value: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
        cout << "x: " << x.get(GRB_DoubleAttr_X) << endl;
        cout << "y: " << y.get(GRB_DoubleAttr_X) << endl;
        cout << "z: " << z.get(GRB_DoubleAttr_X) << endl;

    } catch (GRBException e) {
        cerr << "Error code = " << e.getErrorCode() << endl;
        cerr << e.getMessage() << endl;
    } catch (...) {
        cerr << "Exception during optimization" << endl;
    }
}

// void clustering(rust::Vec<NodeInfo> elements) {
//     vector<NodeInfo> vec(elements.begin(), elements.end());
//     // for (auto& node : vec) {
//     //     cout << "x: " << node.position.x << ", y: " << node.position.y << endl;
//     // }
//     try {
//         // Define points
//         const int num_points = 200;
//         vector<pair<int, int>> points(num_points);
//         for (int i = 0; i < num_points; ++i) {
//             points[i] = {i, i};
//         }

//     } catch (GRBException &e) {
//         cerr << "Error code = " << e.getErrorCode() << endl;
//         cerr << e.getMessage() << endl;
//     } catch (...) {
//         cerr << "Exception during optimization" << endl;
//     }
// }

// void test(rust::Vec<Tuple2_int> gridSize, rust::Vec<Tuple2_int> tiles_arg) {
//     // vector<vector<int>> tiles2(tiles_arg.size());
//     // gridSize.size();
// }
template <typename T>
rust::Vec<T> empty_rust_vec(size_t size) {
    rust::Vec<T> rust_vec;
    rust_vec.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        rust_vec.emplace_back(T());
    }
    return rust_vec;
}

template <typename T>
rust::Vec<T> populate_rust_vec(const vector<T>& cpp_vec) {
    rust::Vec<T> rust_vec;
    rust_vec.reserve(cpp_vec.size());
    for (const auto& value : cpp_vec) {
        rust_vec.emplace_back(value);
    }
    return rust_vec;
}

template <typename T>
bool allEqual(std::initializer_list<T> list) {
    return std::equal(list.begin(), list.end(), list.begin());
}

rust::Vec<SpatialInfo> solveTilingProblem(
    const Tuple2_int gridSize,
    const rust::Vec<TileInfo> tileInfos,
    const rust::Vec<List_int> spatialOccupancy,
    const int split,
    bool output) {
    //"Spatial occupancy size mismatch"
    assert((int(spatialOccupancy.size()) == gridSize.first && int(spatialOccupancy[0].elements.size()) == gridSize.second));
    int tileSizes = tileInfos.size();
    start_env();
    try {
        // Grid size
        int N = gridSize.first;
        int M = gridSize.second;

        // Tile areas
        vector<int> tileAreas(tileSizes);
        ranges::transform(tileInfos, tileAreas.begin(), [](const auto& tile) {
            return tile.size.first * tile.size.second;
        });

        // Create Gurobi model
        if (!output) {
            env.set(GRB_IntParam_OutputFlag, 0);
        }
        // env.set(GRB_IntParam_Threads, 1);
        GRBModel model = GRBModel(env);
        // set nodemethod to 2
        // model.set(GRB_IntParam_NodeMethod, 0);
        // set presolve to 0
        // model.set(GRB_IntParam_Presolve, 2);
        // set MIPFocus to 1
        // model.set(GRB_IntParam_MIPFocus, 1);
        // NoRelHeurTime
        // model.set(GRB_DoubleParam_NoRelHeurTime, 5.0);
        // set PrePasses to 1
        // model.set(GRB_IntParam_PrePasses, 1);
        // set Cuts to 0
        // model.set(GRB_IntParam_Cuts, 0);
        // set Heuristics to 0
        // model.set(GRB_IntParam_Heuristics, 0);
        // set MIPGap to 0
        // model.set(GRB_DoubleParam_MIPGap, 0);

        // Decision variables
        vector<vector<vector<GRBVar>>> x(tileSizes, vector<vector<GRBVar>>(N, vector<GRBVar>(M)));
        vector<vector<GRBVar>> y(N, vector<GRBVar>(M));
        vector<vector<GRBVar>> yWeight(N, vector<GRBVar>(M));

        for (int k = 0; k < tileSizes; ++k) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    x[k][i][j] = model.addVar(0, 1, 0, GRB_BINARY);
                }
            }
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                y[i][j] = model.addVar(0, 1, 0, GRB_BINARY);
                yWeight[i][j] = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
            }
        }

        // split constraints
        for (int k = 0; k < tileSizes; ++k) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    if (i % split != 0) {
                        model.addConstr(x[k][i][j] == 0);
                    }
                }
            }
        }
        // Tile placement constraints
        for (int k = 0; k < tileSizes; ++k) {
            int tileW = tileInfos[k].size.first;
            int tileH = tileInfos[k].size.second;

            for (int i = N - tileW + 1; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    model.addConstr(x[k][i][j] == 0);
                }
            }
            for (int j = M - tileH + 1; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    model.addConstr(x[k][i][j] == 0);
                }
            }
        }

        // Coverage constraints
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                GRBLinExpr coverage = 0;
                GRBLinExpr weightExpr = 0;

                for (int k = 0; k < tileSizes; ++k) {
                    int tileW = tileInfos[k].size.first;
                    int tileH = tileInfos[k].size.second;

                    for (int r = max(0, i - tileW + 1); r <= i; ++r) {
                        for (int c = max(0, j - tileH + 1); c <= j; ++c) {
                            if (r + tileW <= N && c + tileH <= M) {
                                coverage += x[k][r][c];
                                weightExpr += x[k][r][c] * tileInfos[k].weight / tileAreas[k];
                            }
                        }
                    }
                }
                coverage += spatialOccupancy[i].elements[j];
                model.addConstr(y[i][j] == coverage);
                model.addConstr(yWeight[i][j] == weightExpr);
                model.addConstr(y[i][j] <= 1);
            }
        }

        // Tile count limits

        for (int k = 0; k < tileSizes; ++k) {
            if (tileInfos[k].limit < 0) {
                continue;
            }
            GRBLinExpr tileCount = 0;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < M; ++j) {
                    tileCount += x[k][i][j];
                }
            }
            model.addConstr(tileCount <= tileInfos[k].limit);
        }

        // Objective: Maximize total weight coverage
        GRBLinExpr objective = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                objective += yWeight[i][j];
            }
        }
        model.setObjective(objective, GRB_MAXIMIZE);
        // Solve the model
        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            rust::Vec<SpatialInfo> spatialInfoVec = empty_rust_vec<SpatialInfo>(tileSizes);
            for (int k = 0; k < tileSizes; ++k) {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < M; ++j) {
                        if (x[k][i][j].get(GRB_DoubleAttr_X) > 0.5) {
                            spatialInfoVec[k].capacity += 1;
                            spatialInfoVec[k].positions.emplace_back(i, j);
                        }
                    }
                }
            }
            if (output) {
                double totalCoverage = 0;
                for (int k = 0; k < tileSizes; ++k) {
                    totalCoverage += spatialInfoVec[k].capacity * tileInfos[k].size.first * tileInfos[k].size.second;
                }
                print("Optimal objective: " + to_string(model.get(GRB_DoubleAttr_ObjVal)));
                for (int k = 0; k < tileSizes; ++k) {
                    print("Tile type " + to_string(k) + " (" + to_string(tileInfos[k].size.second) + "x" + to_string(tileInfos[k].size.first) + "): " + to_string(spatialInfoVec[k].capacity));
                }
                print("Total coverage: " + to_string(totalCoverage / (N * M)));
            }
            return spatialInfoVec;
        }
        return {};
    } catch (GRBException& e) {
        cerr << "Error code = " << e.getErrorCode() << endl;
        cerr << e.getMessage() << endl;
        print(gridSize.first, gridSize.second);
        print();
        return {};
    } catch (...) {
        cerr << "Exception during optimization." << endl;
        return {};
    }
}

rust::Vec<List_int> solveMultipleKnapsackProblem(
    const rust::Vec<Pair_Int_ListFloat> items,
    const rust::Vec<int> knapsack_capacities) {
    if (items.empty()) {
        return {};
    }

    int total_capacity = 0;
    for (int cap : knapsack_capacities) {
        total_capacity += cap;
    }

    assert(int(items.size()) <= total_capacity && "Not enough knapsacks.");

    start_env();
    try {
        GRBModel model = GRBModel(env);
        // model.set(GRB_IntParam_LogToConsole, 0);

        int num_items = items.size();
        int num_knapsacks = knapsack_capacities.size();

        // Decision variables
        std::vector<std::vector<GRBVar>> x(num_items, std::vector<GRBVar>(num_knapsacks));
        for (int i = 0; i < num_items; ++i) {
            for (int j = 0; j < num_knapsacks; ++j) {
                x[i][j] = model.addVar(0, 1, 0, GRB_BINARY, "x_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }

        // Constraint 1: Each item can be assigned to at most one knapsack
        for (int i = 0; i < num_items; ++i) {
            GRBLinExpr assignment = 0;
            for (int j = 0; j < num_knapsacks; ++j) {
                assignment += x[i][j];
            }
            model.addConstr(assignment == 1, "item_assignment_" + std::to_string(i));
        }

        // Constraint 2: The total weight in each knapsack must not exceed capacity
        for (int j = 0; j < num_knapsacks; ++j) {
            GRBLinExpr total_weight = 0;
            for (int i = 0; i < num_items; ++i) {
                total_weight += x[i][j] * items[i].first;
            }
            model.addConstr(total_weight <= knapsack_capacities[j], "knapsack_capacity_" + std::to_string(j));
        }

        // Objective: Maximize total packed item values
        GRBLinExpr obj = 0;
        for (int i = 0; i < num_items; ++i) {
            for (int j = 0; j < num_knapsacks; ++j) {
                obj += x[i][j] * items[i].second.elements[j];
            }
        }
        model.setObjective(obj, GRB_MAXIMIZE);

        // Optimize the model
        model.optimize();

        // Extract the results
        rust::Vec<List_int> result = empty_rust_vec<List_int>(num_knapsacks);
        // std::vector<std::vector<int>> result(num_knapsacks);
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            for (int i = 0; i < num_items; ++i) {
                bool assigned = false;
                for (int j = 0; j < num_knapsacks; ++j) {
                    if (x[i][j].get(GRB_DoubleAttr_X) > 0.5) {
                        result[j].elements.push_back(i);
                        assigned = true;
                    }
                }
                if (!assigned) {
                    std::cerr << "Item " << i << " is not assigned to any knapsack.\n";
                    throw std::runtime_error("Item not assigned.");
                }
            }
        } else {
            std::cerr << "No feasible solution found.\n";
            throw std::runtime_error("Optimization failed.");
        }

        return result;
    } catch (GRBException& e) {
        std::cerr << "Gurobi error: " << e.getMessage() << std::endl;
        throw;
    }
}

rust::Vec<List_bool> solve_tiling_problem(
    const rust::Vec<List_bool> cover_map,
    const Tuple2_int tile_size) {
    size_t num_row = cover_map.size();
    size_t num_column = cover_map[0].elements.size();
    size_t tile_h = tile_size.first;
    size_t tile_w = tile_size.second;
    rust::Vec<List_bool> result = empty_rust_vec<List_bool>(num_row);
    for (size_t i = 0; i < num_row; ++i) {
        for (size_t j = 0; j < num_column; ++j) {
            result[i].elements.emplace_back(false);
        }
    }
    if (num_column < tile_w || num_row < tile_h) {
        // std::cerr << "Tile size is larger than grid size." << std::endl;
        return result;
    }
    start_env();
    try {
        GRBModel model = GRBModel(env);
        model.set(GRB_IntParam_OutputFlag, 0);  // Disable output
        // model.set(GRB_IntParam_Threads, 1);     // Limit thread
        // Decision variables: x[i][j]
        vector<vector<GRBVar>> x(num_column, vector<GRBVar>(num_row));
        for (size_t i = 0; i < num_column; ++i) {
            for (size_t j = 0; j < num_row; ++j) {
                if (!cover_map[j].elements[i]) {
                    // Only create variables for cells that are covered
                    x[i][j] = model.addVar(0, 1, 0, GRB_CONTINUOUS, "");
                } else {
                    x[i][j] = model.addVar(0, 0, 0, GRB_CONTINUOUS, "");
                }
            }
        }

        // Coverage constraints
        for (size_t i = 0; i <= num_column - tile_w; ++i) {
            for (size_t j = 0; j <= num_row - tile_h; ++j) {
                GRBLinExpr coverage = 0;
                for (size_t r = i; r < i + tile_w; ++r) {
                    for (size_t c = j; c < j + tile_h; ++c) {
                        coverage += x[r][c];
                    }
                }
                model.addConstr(coverage <= 1);
            }
        }

        // Objective: maximize total coverage
        GRBLinExpr obj = 0;
        for (size_t i = 0; i < num_column; ++i)
            for (size_t j = 0; j < num_row; ++j)
                obj += x[i][j];
        model.setObjective(obj, GRB_MAXIMIZE);

        // Solve the model
        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            // double objective_value = model.get(GRB_DoubleAttr_ObjVal);
            // std::cout << "Optimal objective value: " << objective_value << std::endl;
            for (size_t i = 0; i < num_row; i += tile_h) {
                for (size_t j = 0; j < num_column; j += tile_w) {
                    for (size_t r = i; r < std::min(i + tile_h, num_row); ++r) {
                        for (size_t c = j; c < std::min(j + tile_w, num_column); ++c) {
                            if (x[c][r].get(GRB_DoubleAttr_X) > 0.1) {
                                result[i].elements[j] = true;
                                goto next_tile;  // Break nested loops
                            }
                        }
                    }
                next_tile:;
                }
            }
            return result;
        } else {
            throw std::runtime_error("Optimization failed with status: " + std::to_string(model.get(GRB_IntAttr_Status)));
        }
    } catch (GRBException& e) {
        cout << "Gurobi exception: " << e.getMessage() << endl;
        throw std::runtime_error("Gurobi exception: " + std::string(e.getMessage()));
    } catch (std::exception& e) {
        cout << "Exception: " << e.what() << endl;
        throw std::runtime_error("Exception: " + std::string(e.what()));
    }
}

// g++ -std=c++17 hello_world/src/test.cpp -lgurobi_c++ -lgurobi120
// -I/opt/gurobi/gurobi1200/linux64/include -L/opt/gurobi/gurobi1200/linux64/lib
