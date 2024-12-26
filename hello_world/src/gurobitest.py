import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB, Model, quicksum
from scipy.cluster.vq import kmeans
from utility_image_wo_torch import *

# points = [(0, 0), (1, 1), (2, 2), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (1000, 1000)]
np.random.seed(0)
points = [(np.random.random() * 100, np.random.random() * 200) for i in range(200)]
points.sort(key=lambda x: x[0] + x[1])

x = plt.scatter(*zip(*points))
plot_images(seaborn_to_array(x), 500)
# exit()
num_points = len(points)
upper_bound = np.abs(points).sum()
N = 4  # Capacity of each group
K = num_points // N + 1  # Number of clusters
index = list(range(len(points)))
cindex = list(range(K))
model = Model()
model.Params.Presolve = 2
model.Params.TimeLimit = 60
model.Params.MIPFocus = 0
# model.Params.SolutionLimit = 1
# model.Params.MIPGap = 0.5
x = model.addVars(index, cindex, vtype=GRB.BINARY, name="x")
num_select = model.addVars(cindex, lb=1, ub=num_points, vtype=GRB.INTEGER)
# non_empty_col = model.addVars(cindex, vtype=GRB.BINARY)
group_centroid = model.addVars(cindex, [0, 1], lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS)
group_centroid_gap = model.addVars(
    index, cindex, [0, 1], lb=-GRB.INFINITY, ub=upper_bound, vtype=GRB.CONTINUOUS
)
group_centroid_gap_abs = model.addVars(
    index, cindex, [0, 1], lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS
)
group_centroid_gap_abs_sum = model.addVars(cindex, lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS)
for i in index:
    model.addConstr(quicksum(x[i, j] for j in cindex) == 1)
for j in cindex:
    model.addConstr(quicksum(x[i, j] for i in index) <= N)
for j in cindex:
    model.addConstr(num_select[j] == quicksum(x[i, j] for i in index))
    # model.addConstr((non_empty_col[j] == 1) >> (num_select[j] >= 1))
    # model.addConstr((non_empty_col[j] == 0) >> (num_select[j] == 0))

for j in cindex:
    model.addConstr(
        group_centroid[j, 0] * num_select[j] == quicksum(points[i][0] * x[i, j] for i in index)
    )
    model.addConstr(
        group_centroid[j, 1] * num_select[j] == quicksum(points[i][1] * x[i, j] for i in index)
    )
for i in index:
    for j in cindex:

        model.addConstr(
            (x[i, j] == 1) >> (group_centroid_gap[i, j, 0] == (group_centroid[j, 0] - points[i][0]))
        )
        model.addConstr((x[i, j] == 0) >> (group_centroid_gap[i, j, 1] == 0))
        model.addConstr(
            (x[i, j] == 1) >> (group_centroid_gap[i, j, 1] == (group_centroid[j, 1] - points[i][1]))
        )
        model.addConstr((x[i, j] == 0) >> (group_centroid_gap[i, j, 1] == 0))
        model.addConstr(group_centroid_gap_abs[i, j, 0] == gp.abs_(group_centroid_gap[i, j, 0]))
        model.addConstr(group_centroid_gap_abs[i, j, 1] == gp.abs_(group_centroid_gap[i, j, 1]))

for j in cindex:
    model.addConstr(
        group_centroid_gap_abs_sum[j]
        == quicksum(
            (group_centroid_gap_abs[i, j, 0] + group_centroid_gap_abs[i, j, 1]) for i in index
        )
    )
model.optimize()
exit()

# model.setObjectiveN(quicksum(non_empty_col[i] for i in cindex), 0, 1)
# model.setObjectiveN(
#     quicksum(group_centroid_gap_abs_sum[i] for i in cindex),
#     1,
#     0,
# )
# model.setObjective(quicksum(group_centroid_gap_abs_sum[i] for i in cindex), GRB.MINIMIZE)

model._best_obj = None
model._no_improvement_count = 0


def stop_after_no_improvement(model, where):
    if where == GRB.Callback.MIPSOL:
        # Get the current objective value of the new incumbent solution
        current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        print("--------------------")
        print(current_obj)
        # Check if it's the first solution or if improvement occurred
        if model._best_obj is None or current_obj < model._best_obj:
            model._best_obj = current_obj  # Update the best solution found
            model._no_improvement_count = 0  # Reset the counter
        else:
            model._no_improvement_count += 1  # Increment no-improvement count

        # Stop if no improvement after 5 consecutive solutions
        if model._no_improvement_count >= 5:
            print("No improvement after 5 solutions. Stopping early.")
            model.terminate()


model.optimize(lambda m, where: stop_after_no_improvement(m, where))
# model.optimize()

# print("Objective 1 value:", model.getObjective(0).getValue())
# print("Objective 2 value:", model.getObjective(1).getValue())

# if len(points) <= 40:
#     print([non_empty_col[x].x for x in non_empty_col])
#     for i in index:
#         for j in cindex:
#             print("1" if x[i, j].x > 0.5 else "0", end="")
#         print()
#     group_centroids = np.reshape([group_centroid[x].x for x in group_centroid], (-1, 2))
#     print(group_centroids)

# add color based on goup results
colors = np.random.rand(K, 3)
plt.scatter(*zip(*points))
# for i in range(K):
#     plt.scatter(*group_centroid[i], color=colors[i], marker="x")
centers = []
for i, v in list(non_empty_col.items()):
    if v.X < 0.5:
        continue
    for j in index:
        if x[j, i].X > 0.5:
            plt.scatter(*points[j], color=colors[i])
    center = (group_centroid[i, 0].X, group_centroid[i, 1].X)
    plt.scatter(*center, color=colors[i], marker="x")
    # centers.append(center)
# plt.scatter(*zip(*centers), color=colors, marker="x")
plt.show()
num_clusters = len(centers)
# plot_images(seaborn_to_array(plt.gcf()), 500)
kmeans_result = kmeans(points, num_clusters)
plt.scatter(*zip(*points))
for i in range(K):
    plt.scatter(*kmeans_result[0][i], color=colors[i], marker="x")
plt.show()
# plot_images(seaborn_to_array(plt.gcf()), 500)
