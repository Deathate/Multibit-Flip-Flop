// src/my_cpp_code.cpp
#pragma once
#include <iostream>
using namespace std;
#include "/opt/gurobi/gurobi1200/linux64/include/gurobi_c++.h"
#include "hello_world/src/bridge.rs.h"
#include "print.hpp"
#include "rust/cxx.h"
GRBEnv env;
bool env_start = false;

void start_env() {
    if (!env_start) {
        // env.set("LogFile", "gurobi.log");
        env.start();
        env_start = true;
    }
}

void print_message_from_rust(rust::Vec<NodeInfo> elements) {
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

void clustering(rust::Vec<NodeInfo> elements) {
    vector<NodeInfo> vec(elements.begin(), elements.end());
    // for (auto& node : vec) {
    //     cout << "x: " << node.position.x << ", y: " << node.position.y << endl;
    // }
    try {
        // Define points
        const int num_points = 200;
        vector<pair<int, int>> points(num_points);
        for (int i = 0; i < num_points; ++i) {
            points[i] = {i, i};
        }

        // Initialize Gurobi environment and model
        start_env();

        GRBModel model = GRBModel(env);

        // Define variables
        vector<vector<GRBVar>> x(num_points, vector<GRBVar>(num_points));
        vector<GRBVar> num_select(num_points);
        vector<GRBVar> non_empty_col(num_points);
        vector<vector<GRBVar>> group_centroid(num_points, vector<GRBVar>(2));
        vector<vector<GRBVar>> group_centroid_gap(num_points, vector<GRBVar>(2));

        for (int i = 0; i < num_points; ++i) {
            for (int j = 0; j < num_points; ++j) {
                x[i][j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
            }
            num_select[i] = model.addVar(0.0, num_points, 0.0, GRB_INTEGER);
            non_empty_col[i] = model.addVar(0.0, num_points, 0.0, GRB_BINARY);
            group_centroid[i][0] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            group_centroid[i][1] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            group_centroid_gap[i][0] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            group_centroid_gap[i][1] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        }

        // Add constraints
        for (int i = 0; i < num_points; ++i) {
            GRBLinExpr row_sum = 0;
            for (int j = 0; j < num_points; ++j) {
                row_sum += x[i][j];
            }
            model.addConstr(row_sum == 1);
        }

        for (int j = 0; j < num_points; ++j) {
            GRBLinExpr col_sum = 0;
            for (int i = 0; i < num_points; ++i) {
                col_sum += x[i][j];
            }
            model.addConstr(col_sum <= 4);
            model.addConstr(num_select[j] == col_sum);
            model.addGenConstrIndicator(non_empty_col[j], 1, num_select[j] >= 1);
            model.addGenConstrIndicator(non_empty_col[j], 0, num_select[j] == 0);
        }

        for (int j = 0; j < num_points; ++j) {
            GRBLinExpr x_weighted_sum_x = 0;
            GRBLinExpr x_weighted_sum_y = 0;
            for (int i = 0; i < num_points; ++i) {
                x_weighted_sum_x += points[i].first * x[i][j];
                x_weighted_sum_y += points[i].second * x[i][j];
            }
            model.addQConstr(group_centroid[j][0] * num_select[j] == x_weighted_sum_x);
            model.addQConstr(group_centroid[j][1] * num_select[j] == x_weighted_sum_y);
            GRBQuadExpr gap = 0;
            for (int i = 0; i < num_points; ++i) {
                auto dx = (group_centroid[j][0] - points[i].first) * x[i][j];
                auto dy = (group_centroid[j][1] - points[i].second) * x[i][j];
                gap += dx * dx + dy * dy;
            }
            // gap -= (num_points - num_select[j]) * (group_centroid[j][0] + group_centroid[j][1]);
        }

        // Define objectives
        GRBLinExpr objective1 = 0;
        for (int i = 0; i < num_points; ++i) {
            objective1 += non_empty_col[i];
        }
        model.setObjectiveN(objective1, 0, 1);

        GRBLinExpr objective2 = 0;
        for (int i = 0; i < num_points; ++i) {
            objective2 += group_centroid[i][0] + group_centroid[i][1];
        }
        model.setObjectiveN(objective2, 1, 0);

        // // Optimize model
        // model.optimize();

        // // Output results
        // cout << "Objective 1 value: " << model.getObjective(0).getValue() << endl;
        // cout << "Objective 2 value: " << model.getObjective(1).getValue() << endl;

        // if (num_points <= 10) {
        //     for (int j = 0; j < num_points; ++j) {
        //         cout << non_empty_col[j].get(GRB_DoubleAttr_X) << " ";
        //     }
        //     cout << endl;

        //     for (int i = 0; i < num_points; ++i) {
        //         for (int j = 0; j < num_points; ++j) {
        //             cout << (x[i][j].get(GRB_DoubleAttr_X) > 0.5 ? "1" : "0");
        //         }
        //         cout << endl;
        //     }

        //     for (int j = 0; j < num_points; ++j) {
        //         cout << "Centroid " << j << ": (" << group_centroid[j][0].get(GRB_DoubleAttr_X) << ", " << group_centroid[j][1].get(GRB_DoubleAttr_X) << ")" << endl;
        //     }
        // }

    } catch (GRBException& e) {
        cerr << "Error code = " << e.getErrorCode() << endl;
        cerr << e.getMessage() << endl;
    } catch (...) {
        cerr << "Exception during optimization" << endl;
    }
}

// g++ -std=c++17 hello_world/src/test.cpp -lgurobi_c++ -lgurobi120
// -I/opt/gurobi/gurobi1200/linux64/include -L/opt/gurobi/gurobi1200/linux64/lib
