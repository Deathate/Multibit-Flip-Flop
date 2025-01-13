#include <iostream>
#include <ranges>
// #define NDEBUG
#include <assert.h>

#include <thread>

#include "/opt/gurobi/gurobi1200/linux64/include/gurobi_c++.h"
#include "bridge.h"
#include "formatter.cpp"
#include "print.hpp"
using namespace std;

GRBEnv env;

void start_env() {
    bool env_start = false;
    if (!env_start) {
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
    bool output) {
    //"Spatial occupancy size mismatch"
    assert((spatialOccupancy.size() == gridSize.first && spatialOccupancy[0].elements.size() == gridSize.second));
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
        if (!output) env.set(GRB_IntParam_OutputFlag, 0);
        env.set(GRB_IntParam_Threads, 1);
        GRBModel model = GRBModel(env);

        // Decision variables
        vector<vector<vector<GRBVar>>> x(tileSizes, vector<vector<GRBVar>>(N, vector<GRBVar>(M)));
        vector<vector<GRBVar>> y(N, vector<GRBVar>(M));
        vector<vector<GRBVar>> yWeight(N, vector<GRBVar>(M));

        for (size_t k = 0; k < tileSizes; ++k) {
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

        // Tile placement constraints
        for (size_t k = 0; k < tileSizes; ++k) {
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

                for (size_t k = 0; k < tileSizes; ++k) {
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

        for (size_t k = 0; k < tileSizes; ++k) {
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
            for (size_t k = 0; k < tileSizes; ++k) {
                spatialInfoVec[k].bits = tileInfos[k].bits;
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < M; ++j) {
                        if (x[k][i][j].get(GRB_DoubleAttr_X) > 0.5) {
                            spatialInfoVec[k].capacity += 1;
                        }
                    }
                }
            }
            if (output) {
                double totalCoverage = 0;
                for (size_t k = 0; k < tileSizes; ++k) {
                    totalCoverage += spatialInfoVec[k].capacity * tileInfos[k].size.first * tileInfos[k].size.second;
                }
                print("Optimal objective: " + to_string(model.get(GRB_DoubleAttr_ObjVal)));
                for (size_t k = 0; k < tileSizes; ++k) {
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

// g++ -std=c++17 hello_world/src/test.cpp -lgurobi_c++ -lgurobi120
// -I/opt/gurobi/gurobi1200/linux64/include -L/opt/gurobi/gurobi1200/linux64/lib
