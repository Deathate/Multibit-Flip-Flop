# %%
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from plot import *
from utility import *

# m = 100000
# n=10000
# # Define the data for the problem
# # supply = [20, 30]  # Supply at each source
# supply = np.random.multinomial(m, np.ones(n)/n)
# # demand = [10, 15, 25]  # Demand at each destination
# demand =np.random.multinomial(m, np.ones(n)/n)
# # cost = [[10, 15, 20],  # Cost matrix for transporting from sources to destinations
# #         [12, 14, 16]]
# cost = np.random.random((n,n))

# # Create a new model
# model = gp.Model("OTP")

# # Define decision variables
# num_sources = len(supply)
# num_destinations = len(demand)
# x = {}  # Transportation variables
# for i in range(num_sources):
#     for j in range(num_destinations):
#         x[i, j] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
# # Set objective function: minimize total transportation cost
# model.setObjective(gp.quicksum(cost[i][j] * x[i, j] for i in range(num_sources) for j in range(num_destinations)), GRB.MINIMIZE)

# # Add supply constraints
# for i in range(num_sources):
#     model.addConstr(gp.quicksum(x[i, j] for j in range(num_destinations)) <= supply[i], f"supply_{i}")

# # Add demand constraints
# for j in range(num_destinations):
#     model.addConstr(gp.quicksum(x[i, j] for i in range(num_sources)) >= demand[j], f"demand_{j}")

# # Optimize the model
# model.optimize()

# # Print the optimal solution
# if model.status == GRB.OPTIMAL:
#     print("Optimal solution found:")
#     for i in range(num_sources):
#         for j in range(num_destinations):
#             print(f"Transport {supply[i]} units from source {i} to destination {j}: {x[i, j].x}")
a = BoxContainer(1, offset=[1, 0])
b = BoxContainer(2, offset=[5, 3])
c = BoxContainer(2, offset=[6, 2])
d = BoxContainer(1.5, offset=[2, 2])
P = PlotlyUtility()
P.add_rectangle(a)
P.add_rectangle(b)
P.add_rectangle(c)
P.add_rectangle(d)
P.show()
