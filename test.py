import gurobipy as gp
from gurobipy import GRB

model = gp.Model("tiling")
a=[]
for i in range(100):
    a.append(model.addVar())
    model.addConstr(a[i] == i)
x = model.addVar()
model.addConstr(x == 520)
def add_dis_index_var(model, vars):
    L = len(vars)
    c = model.addVars(L, lb=-GRB.INFINITY)
    d = model.addVars(L)
    e = model.addVar()
    cv = model.addVars(L, vtype=GRB.BINARY)
    idx = model.addVar(vtype=GRB.INTEGER)
    model.addConstr(e == gp.min_(d))
    for i in range(L):
        model.addConstr(c[i] == x - vars[i])
        model.addConstr(d[i] == gp.abs_(c[i]))
        model.addConstr((cv[i] == 1) >> (d[i] == e))
        model.addConstr((cv[i] == 1) >> (idx == i))
    model.addConstr(cv[idx] == 0)
    model.addConstr(gp.quicksum(cv) == 1)
    return idx
idx = add_dis_index_var(model, a)
# Optimize the model
model.optimize()
if model.status == GRB.OPTIMAL:
    # print(x.X)
    # print(c[0].X)
    print(idx)
# Print the solution
# if model.status == GRB.OPTIMAL:
#     for k in range(grid_width):
#         for l in range(grid_height):
#             for i in range(grid_width):
#                 for j in range(grid_height):
#                     if x[i, j, k, l].X > 0.5:
#                         print(f"Tile placed at ({k},{l}) covering ({i},{j})")
# else:
#     print("No optimal solution found")
