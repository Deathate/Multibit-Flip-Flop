# %%
# import gurobipy as gp
# from gurobipy import GRB

# model = gp.Model("tiling")
# a = []
# for i in range(100):
#     a.append(model.addVar())
#     model.addConstr(a[i] == i)
# x = model.addVar()
# model.addConstr(x == 520)


# def add_dis_index_var(model, vars):
#     L = len(vars)
#     c = model.addVars(L, lb=-GRB.INFINITY)
#     d = model.addVars(L)
#     e = model.addVar()
#     cv = model.addVars(L, vtype=GRB.BINARY)
#     idx = model.addVar(vtype=GRB.INTEGER)
#     model.addConstr(e == gp.min_(d))
#     for i in range(L):
#         model.addConstr(c[i] == x - vars[i])
#         model.addConstr(d[i] == gp.abs_(c[i]))
#         model.addConstr((cv[i] == 1) >> (d[i] == e))
#         model.addConstr((cv[i] == 1) >> (idx == i))
#     model.addConstr(cv[idx] == 0)
#     model.addConstr(gp.quicksum(cv) == 1)
#     return idx


# idx = add_dis_index_var(model, a)
# # Optimize the model
# model.optimize()
# if model.status == GRB.OPTIMAL:
#     # print(x.X)
#     # print(c[0].X)
#     print(idx)
# # Print the solution
# # if model.status == GRB.OPTIMAL:
# #     for k in range(grid_width):
# #         for l in range(grid_height):
# #             for i in range(grid_width):
# #                 for j in range(grid_height):
# #                     if x[i, j, k, l].X > 0.5:
# #                         print(f"Tile placed at ({k},{l}) covering ({i},{j})")
# # else:
# #     print("No optimal solution found")
import random

import gurobipy as gp
from gurobipy import GRB, Model, quicksum

# Generate random data for 200 factories and 100 warehouses
num_factories = 10
num_warehouses = 10

factories = [f"F{i+1}" for i in range(num_factories)]
warehouses = [f"W{j+1}" for j in range(num_warehouses)]

# Random supply for each factory (e.g., between 50 and 500 units)
supply = {f: random.randint(50, 500) for f in factories}

# Random demand for each warehouse (e.g., between 30 and 300 units)
demand = {w: random.randint(30, 300) for w in warehouses}

# Random transportation cost between factories and warehouses (e.g., between 1 and 20 per unit)
# transport_cost = {(f, w): random.randint(1, 20) for f in factories for w in warehouses}

# Model
model = Model("Factory_Warehouse_Optimization")

# Variables
# x[f, w]: Binary variable, 1 if factory f supplies to warehouse w, 0 otherwise
x = model.addVars(factories, warehouses, vtype=GRB.BINARY, name="x")
y = model.addVars(warehouses, vtype=GRB.CONTINUOUS, name="y")
y_abs = model.addVars(warehouses, vtype=GRB.CONTINUOUS, name="y_abs")
y_c = model.addVars(warehouses, vtype=GRB.INTEGER, name="y_c")
y_d = model.addVars(warehouses, vtype=GRB.BINARY, name="y_c")
for w in warehouses:
    model.addConstr(y_c[w] == quicksum(x[f, w] for f in factories))
    model.addConstr((y_d[w] == 1) >> (y_c[w] >= 1))
    model.addConstr((y_d[w] == 0) >> (y_c[w] == 0))
    model.addConstr((y[w] - demand[w]) * y_c[w] == quicksum(supply[f] * x[f, w] for f in factories))
    model.addConstr((y_d[w] == 0) >> (y[w] == 0))
    model.addConstr(y_abs[w] == gp.abs_(y[w]))
# Objective: Minimize total transportation cost
model.setObjective(quicksum(y_abs[w] for w in warehouses), GRB.MINIMIZE)

# Constraints: Each factory can supply to only one warehouse
for f in factories:
    model.addConstr(quicksum(x[f, w] for w in warehouses) == 1, name=f"Factory_{f}_One_Warehouse")

# Constraints: Each warehouse can accept goods from up to two factories
for w in warehouses:
    model.addConstr(quicksum(x[f, w] for f in factories) <= 4, name=f"Warehouse_{w}_Two_Factories")

# Solve the model
model.optimize()

# Output the results
print(supply)
print(demand)
if model.status == GRB.OPTIMAL:
    print(f"Optimal Total Cost: {model.objVal}")
    for f in factories:
        for w in warehouses:
            if x[f, w].x > 0.5:  # Binary variable, check if it's 1
                print(f"Factory {f} supplies to Warehouse {w}")
    for w in warehouses:
        score = sum(supply[f] for f in factories if x[f, w].x > 0.5) / max(y_c[w].x, 1) - demand[w]
        print(f"Warehouse {w} accepts from {y_c[w].x} factories, score: {score}")
        print(y_abs[w].x)
        # print(sum(supply[f] * x[f, w].x for f in factories))
        print(demand[w])

else:
    print("No optimal solution found.")
import time

# %%
import gurobipy as gp
import numpy as np
from gurobipy import GRB, Model, quicksum

# points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
points = [(i, i) for i in range(50)]
num_points = len(points)
upper_bound = np.abs(points).sum()
N = 4  # Capacity of each group
K = num_points // N + 1  # Number of clusters
index = list(range(len(points)))
cindex = list(range(K))
model = Model()
model.Params.Presolve = 2
x = model.addVars(index, cindex, vtype=GRB.BINARY, name="x")
num_select = model.addVars(cindex, lb=0, ub=num_points, vtype=GRB.INTEGER)
non_empty_col = model.addVars(cindex, vtype=GRB.BINARY)
group_centroid = model.addVars(cindex, [0, 1], vtype=GRB.CONTINUOUS)
group_centroid_gap = model.addVars(index, cindex, [0, 1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
group_centroid_gap_abs = model.addVars(index, cindex, [0, 1], lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS)
group_centroid_gap_abs_sum = model.addVars(cindex, lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS)
for i in index:
    model.addConstr(quicksum(x[i, j] for j in cindex) == 1)
for j in cindex:
    model.addConstr(quicksum(x[i, j] for i in index) <= N)
for j in cindex:
    model.addConstr(num_select[j] == quicksum(x[i, j] for i in index))
    model.addConstr((non_empty_col[j] == 1) >> (num_select[j] >= 1))
    model.addConstr((non_empty_col[j] == 0) >> (num_select[j] == 0))

for j in cindex:
    model.addConstr(
        group_centroid[j, 0] * num_select[j] == quicksum(points[i][0] * x[i, j] for i in index)
    )
    model.addConstr(
        group_centroid[j, 1] * num_select[j] == quicksum(points[i][1] * x[i, j] for i in index)
    )

for i in index:
    for j in cindex:
        model.addConstr(group_centroid_gap[i, j, 0] == (group_centroid[j, 0] - points[i][0]))
        model.addConstr(group_centroid_gap[i, j, 1] == (group_centroid[j, 1] - points[i][1]))
        model.addConstr(group_centroid_gap_abs[i, j, 0] == gp.abs_(group_centroid_gap[i, j, 0]))
        model.addConstr(group_centroid_gap_abs[i, j, 1] == gp.abs_(group_centroid_gap[i, j, 1]))

for j in cindex:
    model.addConstr(
        group_centroid_gap_abs_sum[j]
        == quicksum(
            (group_centroid_gap_abs[i, j, 0] + group_centroid_gap_abs[i, j, 1]) * x[i, j]
            for i in index
        )
    )

# model.setObjectiveN(quicksum(non_empty_col[i] for i in cindex), 0, 1)
# model.setObjectiveN(
#     quicksum(group_centroid_gap_abs_sum[i] for i in cindex),
#     1,
#     0,
# )
model.setObjective(quicksum(group_centroid_gap_abs_sum[i] for i in cindex), GRB.MINIMIZE)
model.optimize()

# print("Objective 1 value:", model.getObjective(0).getValue())
# print("Objective 2 value:", model.getObjective(1).getValue())

if len(points) <= 10:
    print([non_empty_col[x].x for x in non_empty_col])
    for i in index:
        for j in cindex:
            print("1" if x[i, j].x > 0.5 else "0", end="")
        print()
    group_centroids = np.reshape([group_centroid[x].x for x in group_centroid], (-1, 2))
    print(group_centroids)
# print(group_centroids.sum())
