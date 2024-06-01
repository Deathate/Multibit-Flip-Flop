import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("conditional_logic_LP")


xy = model.addMVar((2,), vtype=GRB.BINARY, name="xy")


def cityblock_variable(model, v1, v2, bias):
    delta_xy = model.addMVar(2,lb=-GRB.INFINITY, name="delta_xy")
    abs_delta_xy = model.addMVar(2)
    cityblock_distance = model.addVar(lb=-GRB.INFINITY, name="cityblock_distance")
    model.addConstr(delta_xy[0] == v1[0] - v2[0])
    model.addConstr(delta_xy[1] == v1[1] - v2[1])
    model.addConstr(abs_delta_xy[0] == gp.abs_(delta_xy[0]))
    model.addConstr(abs_delta_xy[1] == gp.abs_(delta_xy[1]))
    model.addConstr(cityblock_distance == gp.quicksum(abs_delta_xy) + bias)
    return cityblock_distance


prev_pin_displacement_delay = cityblock_variable(model, xy, [0, 25], 13)
# Optimize model
model.optimize()

# Display the results
# if model.status == GRB.OPTIMAL:
#     print(f")
# else:
#     print("No optimal solution found.")
