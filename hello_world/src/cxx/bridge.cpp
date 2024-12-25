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

void print_message_from_rust(rust::Vec<NodeInfo> elements) {
    try {
        // Create environment
        if (!env_start) {
            env.set("LogFile", "gurobi.log");
            env.start();
            env_start = true;
        }

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

// }
// g++ -std=c++17 hello_world/src/test.cpp -lgurobi_c++ -lgurobi120
// -I/opt/gurobi/gurobi1200/linux64/include -L/opt/gurobi/gurobi1200/linux64/lib
