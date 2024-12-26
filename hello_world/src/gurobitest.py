import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB, Model, quicksum
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from utility_image_wo_torch import *

# points = [(0, 0), (1, 1), (2, 2), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (1000, 1000)]
np.random.seed(0)
points = [(np.random.random() * 100, np.random.random() * 200) for i in range(200)]
points.sort(key=lambda x: x[0] + x[1])
points = np.array(points)

plt.scatter(*zip(*points))
plot_images(plt.gca(), 500)
# exit()
num_points = len(points)
upper_bound = np.abs(points).sum()
N = 4  # Capacity of each group
K = num_points // N + 1  # Number of clusters
colors = np.random.rand(K, 3)


def kmean_plot():
    plt.figure()
    km = KMeans(n_clusters=K).fit(points)
    labels = km.labels_
    for i, point in enumerate(points):
        plt.scatter(*point, color=colors[labels[i]])
    plt.scatter(*zip(*km.cluster_centers_), color=colors, marker="x")
    plot_images(plt.gcf(), 500)
    centers = km.cluster_centers_
    cluster_sizes = np.bincount(labels)
    for cluster_id in range(max(labels) + 1):
        while cluster_sizes[cluster_id] > 4:
            # Find the farthest point from the cluster center
            cluster_points = np.where(labels == cluster_id)[0]
            distances = np.linalg.norm(points[cluster_points] - centers[cluster_id], axis=1)
            farthest_point_idx = cluster_points[np.argmax(distances)]

            # Reassign this point to the nearest other cluster
            distances_to_centers = np.linalg.norm(points[farthest_point_idx] - centers, axis=1)
            # print(distances_to_centers)
            # print(cluster_sizes)
            # print(np.where(cluster_sizes >= 4))
            # exit()
            distances_to_centers[np.where(cluster_sizes >= 4)] = (
                np.inf
            )  # Exclude the current cluster
            new_cluster_id = np.argmin(distances_to_centers)

            # Update labels and cluster sizes
            labels[farthest_point_idx] = new_cluster_id
            cluster_sizes[cluster_id] -= 1
            cluster_sizes[new_cluster_id] += 1
    plt.figure()
    for i, point in enumerate(points):
        plt.scatter(*point, color=colors[labels[i]])
    plt.scatter(*zip(*km.cluster_centers_), color=colors, marker="x")
    plot_images(plt.gcf(), 500)
    obj = 0
    for i, point in enumerate(points):
        obj += np.linalg.norm(np.array(point) - np.array(km.cluster_centers_[km.labels_[i]]))
    print(obj)
    return km


km = kmean_plot()
print(np.bincount(km.labels_))
exit()
index = list(range(len(points)))
cindex = list(range(K))
model = Model()
model.Params.Presolve = 2
model.Params.TimeLimit = 60
# model.Params.MIPFocus = 0
model.Params.LogToConsole = 0
# model.Params.SolutionLimit = 1
# model.Params.MIPGap = 0.5
x = model.addVars(index, cindex, vtype=GRB.BINARY, name="x")
num_select = model.addVars(cindex, lb=1, ub=num_points, vtype=GRB.INTEGER)
# non_empty_col = model.addVars(cindex, vtype=GRB.BINARY)
group_centroid = model.addVars(cindex, [0, 1], lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS)
group_centroid_gap = model.addVars(index, cindex, [0, 1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
group_centroid_gap_abs = model.addVars(index, cindex, lb=0, vtype=GRB.CONTINUOUS)
# warm start
for i in index:
    for j in cindex:
        x[i, j].start = 0
    label = km.labels_[i]
    x[i, label].start = 1

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
            (group_centroid_gap[i, j, 0] == (group_centroid[j, 0] - points[i][0]) * x[i, j])
        )
        model.addConstr(
            (group_centroid_gap[i, j, 1] == (group_centroid[j, 1] - points[i][1]) * x[i, j])
        )
        model.addConstr(
            (group_centroid_gap_abs[i, j] * group_centroid_gap_abs[i, j])
            == (
                group_centroid_gap[i, j, 0] * group_centroid_gap[i, j, 0]
                + group_centroid_gap[i, j, 1] * group_centroid_gap[i, j, 1]
            )
        )


model.setObjective(
    quicksum(quicksum((group_centroid_gap_abs[i, j]) for i in index) for j in cindex),
    GRB.MINIMIZE,
)

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

if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    plt.scatter(*zip(*points))
    plt.figure()
    centers = []
    obj = 0
    for i, v in list(num_select.items()):
        if v.X == 0:
            continue
        center = (group_centroid[i, 0].X, group_centroid[i, 1].X)
        for j in index:
            if x[j, i].X > 0.5:
                plt.scatter(*points[j], color=colors[i])
                obj += np.linalg.norm(np.array(points[j]) - np.array(center))
        plt.scatter(*center, color=colors[i], marker="x")
    plot_images(plt.gcf(), 500)
    print(obj)

# kmean_plot()
