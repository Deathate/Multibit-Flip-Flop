import time
from types import SimpleNamespace

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from gurobipy import GRB, Model, quicksum
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from tqdm import tqdm
from utility_image_wo_torch import *


def kmean_alg(method, points, N):
    K = len(points) // N + 1  # Number of clusters
    plt.figure()
    km = KMeans(n_clusters=K).fit(points)
    labels = km.labels_
    centers = km.cluster_centers_
    cluster_sizes = np.bincount(labels)
    if method == 0:
        for cluster_id in range(max(labels) + 1):
            while cluster_sizes[cluster_id] > 4:
                # Find the farthest point from the cluster center
                cluster_points = np.where(labels == cluster_id)[0]
                distances = np.linalg.norm(points[cluster_points] - centers[cluster_id], axis=1)
                farthest_point_idx = cluster_points[np.argmax(distances)]

                # Reassign this point to the nearest other cluster
                distances_to_centers = np.linalg.norm(points[farthest_point_idx] - centers, axis=1)
                distances_to_centers[np.where(cluster_sizes >= 4)] = np.inf
                new_cluster_id = np.argmin(distances_to_centers)

                # Update labels and cluster sizes
                labels[farthest_point_idx] = new_cluster_id
                cluster_sizes[cluster_id] -= 1
                cluster_sizes[new_cluster_id] += 1
    elif method == 1:
        walked_id = []
        while True:
            cluster_id = next((i for i in range(K) if cluster_sizes[i] > 4), None)
            if cluster_id is None:
                break
            walked_id.append(cluster_id)
            while cluster_sizes[cluster_id] > 4:
                # print("--")
                # print(cluster_sizes[cluster_id])
                # Find the farthest point from the cluster center
                cluster_points = np.where(labels == cluster_id)[0]
                distances = np.linalg.norm(points[cluster_points] - centers[cluster_id], axis=1)
                farthest_point_idx = cluster_points[np.argmax(distances)]

                # Reassign this point to the nearest other cluster
                distances_to_centers = np.linalg.norm(points[farthest_point_idx] - centers, axis=1)
                distances_to_centers[walked_id] = np.inf
                # print(walked_id)
                new_cluster_id = np.argmin(distances_to_centers)
                # print(new_cluster_id)
                # Update labels and cluster sizes
                labels[farthest_point_idx] = new_cluster_id
                cluster_sizes[cluster_id] -= 1
                cluster_sizes[new_cluster_id] += 1
    elif method == 2:
        walked_id = []
        while True:
            cluster_id = next((i for i in range(K) if cluster_sizes[i] > 4), None)
            if cluster_id is None:
                break
            walked_id.append(cluster_id)
            while cluster_sizes[cluster_id] > 4:
                cluster_points = np.where(labels == cluster_id)[0]
                distances = scipy.spatial.distance.cdist(points[cluster_points], centers)
                distances[:, walked_id] = np.inf
                selected_idx, new_cluster_id = np.unravel_index(
                    np.argmin(distances), distances.shape
                )
                cheapest_point_idx = cluster_points[selected_idx]

                # Update labels and cluster sizes
                labels[cheapest_point_idx] = new_cluster_id
                cluster_sizes[cluster_id] -= 1
                cluster_sizes[new_cluster_id] += 1
    for i in range(len(km.cluster_centers_)):
        km.cluster_centers_[i] = np.mean(points[np.where(labels == i)], axis=0)

    return km


def draw(
    km,
    colors,
):
    plt.figure()
    for i, point in enumerate(points):
        plt.scatter(*point, color=colors[km.labels_[i]])
    plt.scatter(*zip(*km.cluster_centers_), color=colors, marker="x")
    plot_images(plt.gcf(), 500)


def lp_plot(points, N, method, km):
    M = np.abs(points).sum()
    num_points = len(points)
    K = num_points // N + 1  # Number of clusters
    # colors = np.random.rand(K, 3)
    index = list(range(len(points)))
    cindex = list(range(K))
    with Model() as model:
        try:
            model.Params.Presolve = 2
            # model.setParam("LogFile", "gurobi.log")
            model.Params.TimeLimit = 60
            # model.Params.ScaleFlag = 1
            # model.Params.OutputFlag = 0
            # model.Params.MIPFocus = 0
            # model.Params.LogToConsole = 0
            # model.Params.SolutionLimit = 1
            # model.Params.MIPGap = 0.5
            x = model.addVars(index, cindex, vtype=GRB.BINARY, name="x")
            num_select = model.addVars(cindex, ub=N, vtype=GRB.INTEGER)
            # selected = model.addVars(cindex, vtype=GRB.BINARY)
            # non_empty_col = model.addVars(cindex, vtype=GRB.BINARY)
            group_centroid = model.addVars(cindex, [0, 1], vtype=GRB.CONTINUOUS)
            group_centroid_gap = model.addVars(
                index, cindex, [0, 1], lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS
            )
            gap_M = np.linalg.norm(np.max(points, axis=0)) * 10
            group_centroid_gap_abs = model.addVars(index, cindex, vtype=GRB.CONTINUOUS, ub=gap_M)
            # min_val = model.addVars(cindex, vtype=GRB.CONTINUOUS)
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
                # model.addConstr(min_val[j] == gp.min_(x[i, j] for i in index))
            for j in cindex:
                model.addConstr(num_select[j] == quicksum(x[i, j] for i in index))
                # model.addConstr(num_select[j] - N * (1 - selected[j]) >= 1)
            # for j in cindex:
            #     for i in index:
            #         model.addConstr(min_val[j] <= points[i, j] + (1 - x[i, j]) * M)
            #         model.addConstr(min_val[j] >= points[i, j] - (1 - x[i, j]) * M)
            for j in cindex:
                # model.addConstr((selected[j] == 0) >> (group_centroid[j, 0] == 0))
                # model.addConstr((selected[j] == 0) >> (group_centroid[j, 1] == 0))
                model.addConstr(
                    group_centroid[j, 0] * num_select[j]
                    == quicksum(points[i][0] * x[i, j] for i in index)
                )
                model.addConstr(
                    group_centroid[j, 1] * num_select[j]
                    == quicksum(points[i][1] * x[i, j] for i in index)
                )
            if method == 1:
                for i in index:
                    for j in cindex:
                        model.addConstr(
                            (group_centroid_gap[i, j, 0] == (group_centroid[j, 0] - points[i][0]))
                        )
                        model.addConstr(
                            (group_centroid_gap[i, j, 1] == (group_centroid[j, 1] - points[i][1]))
                        )

                        model.addConstr(
                            (group_centroid_gap_abs[i, j] * group_centroid_gap_abs[i, j])
                            == (
                                group_centroid_gap[i, j, 0] * group_centroid_gap[i, j, 0]
                                + group_centroid_gap[i, j, 1] * group_centroid_gap[i, j, 1]
                            )
                        )

                model.setObjective(
                    quicksum(
                        quicksum((group_centroid_gap_abs[i, j] * x[i, j]) for i in index)
                        for j in cindex
                    ),
                    GRB.MINIMIZE,
                )
            elif method == 2:
                for i in index:
                    for j in cindex:
                        model.addConstr(
                            (
                                group_centroid_gap[i, j, 0]
                                == (group_centroid[j, 0] - points[i][0]) * x[i, j]
                            )
                        )
                        model.addConstr(
                            (
                                group_centroid_gap[i, j, 1]
                                == (group_centroid[j, 1] - points[i][1]) * x[i, j]
                            )
                        )

                        model.addConstr(
                            (group_centroid_gap_abs[i, j] * group_centroid_gap_abs[i, j])
                            == (
                                group_centroid_gap[i, j, 0] * group_centroid_gap[i, j, 0]
                                + group_centroid_gap[i, j, 1] * group_centroid_gap[i, j, 1]
                            )
                        )

                model.setObjective(
                    quicksum(
                        quicksum((group_centroid_gap_abs[i, j]) for i in index) for j in cindex
                    ),
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

            # model.optimize(lambda m, where: stop_after_no_improvement(m, where))
            model.optimize()
        except gp.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))
        except Exception as e:
            pass

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.SolCount > 0:
            plt.scatter(*zip(*points))
            plt.figure()
            # LP obj value
            # lp_obj = 0
            # for i, v in list(num_select.items()):
            #     if v.X == 0:
            #         continue
            #     center = (group_centroid[i, 0].X, group_centroid[i, 1].X)
            #     for j in index:
            #         if x[j, i].X > 0.5:
            #             plt.scatter(*points[j], color=colors[i])
            #             lp_obj += np.linalg.norm(np.array(points[j]) - np.array(center))
            #     plt.scatter(*center, color=colors[i], marker="x")
            # plot_images(plt.gcf(), 500)
            labels = []
            for i in index:
                for j in cindex:
                    if x[i, j].X > 0.5:
                        labels.append(j)
                        break
            centers = []
            for i in cindex:
                centers.append(np.mean(points[np.where(np.array(labels) == i)], axis=0))
            lp_results = SimpleNamespace()
            lp_results.cluster_centers_ = centers
            lp_results.labels_ = labels
            return lp_results


def evaluate(points, km):
    km_obj = 0
    for i, point in enumerate(points):
        km_obj += np.linalg.norm(np.array(point) - np.array(km.cluster_centers_[km.labels_[i]]))
    print("objective value:", km_obj)
    return km_obj


# def test(seed, num_points):
#     np.random.seed(seed)
#     points = [(np.random.random() * 100, np.random.random() * 200) for _ in range(num_points)]
#     points.sort(key=lambda x: x[0] + x[1])
#     points = np.array(points)

#     upper_bound = np.abs(points).sum()
#     N = 4  # Capacity of each group
#     K = num_points // N + 1  # Number of clusters
#     colors = np.random.rand(K, 3)

#     km = kmean_plot(method=1, points=points)
#     index = list(range(len(points)))
#     cindex = list(range(K))
#     with Model() as model:
#         # model.Params.Presolve = 2
#         # model.setParam("LogFile", "gurobi.log")
#         model.Params.TimeLimit = 60
#         # model.Params.MIPFocus = 0
#         # model.Params.LogToConsole = 0
#         # model.Params.SolutionLimit = 1
#         # model.Params.MIPGap = 0.5
#         x = model.addVars(index, cindex, vtype=GRB.BINARY, name="x")
#         num_select = model.addVars(cindex, lb=1, ub=N, vtype=GRB.INTEGER)
#         selected = model.addVars(cindex, vtype=GRB.BINARY)
#         # non_empty_col = model.addVars(cindex, vtype=GRB.BINARY)
#         group_centroid = model.addVars(cindex, [0, 1], lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS)
#         group_centroid_gap = model.addVars(
#             index, cindex, [0, 1], lb=-upper_bound, ub=upper_bound, vtype=GRB.CONTINUOUS
#         )
#         group_centroid_gap_abs = model.addVars(
#             index, cindex, lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS
#         )
#         # warm start
#         for i in index:
#             for j in cindex:
#                 x[i, j].start = 0
#             label = km.labels_[i]
#             x[i, label].start = 1

#         for i in index:
#             model.addConstr(quicksum(x[i, j] for j in cindex) == 1)
#         for j in cindex:
#             model.addConstr(quicksum(x[i, j] for i in index) <= N)
#         for j in cindex:
#             model.addConstr(num_select[j] == quicksum(x[i, j] for i in index))
#             # model.addConstr(selected[j] == (num_select[j] >= 1))
#             model.addConstr(num_select[j] - N * (1 - selected[j]) >= 1)
#         for j in cindex:
#             # model.addConstr((selected[j] == 0) >> (group_centroid[j, 0] == 0))
#             # model.addConstr((selected[j] == 0) >> (group_centroid[j, 1] == 0))
#             model.addConstr(
#                 group_centroid[j, 0] * num_select[j]
#                 == quicksum(points[i][0] * x[i, j] for i in index)
#             )
#             model.addConstr(
#                 group_centroid[j, 1] * num_select[j]
#                 == quicksum(points[i][1] * x[i, j] for i in index)
#             )
#         for i in index:
#             for j in cindex:
#                 model.addConstr(
#                     (group_centroid_gap[i, j, 0] == (group_centroid[j, 0] - points[i][0]) * x[i, j])
#                 )
#                 model.addConstr(
#                     (group_centroid_gap[i, j, 1] == (group_centroid[j, 1] - points[i][1]) * x[i, j])
#                 )
#                 model.addConstr(
#                     (group_centroid_gap_abs[i, j] * group_centroid_gap_abs[i, j])
#                     == (
#                         group_centroid_gap[i, j, 0] * group_centroid_gap[i, j, 0]
#                         + group_centroid_gap[i, j, 1] * group_centroid_gap[i, j, 1]
#                     )
#                 )

#         model.setObjective(
#             quicksum(quicksum((group_centroid_gap_abs[i, j]) for i in index) for j in cindex),
#             GRB.MINIMIZE,
#         )

#         model._best_obj = None
#         model._no_improvement_count = 0

#         def stop_after_no_improvement(model, where):
#             if where == GRB.Callback.MIPSOL:
#                 # Get the current objective value of the new incumbent solution
#                 current_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
#                 print("--------------------")
#                 print(current_obj)
#                 # Check if it's the first solution or if improvement occurred
#                 if model._best_obj is None or current_obj < model._best_obj:
#                     model._best_obj = current_obj  # Update the best solution found
#                     model._no_improvement_count = 0  # Reset the counter
#                 else:
#                     model._no_improvement_count += 1  # Increment no-improvement count

#                 # Stop if no improvement after 5 consecutive solutions
#                 if model._no_improvement_count >= 5:
#                     print("No improvement after 5 solutions. Stopping early.")
#                     model.terminate()

#         # model.optimize(lambda m, where: stop_after_no_improvement(m, where))
#         model.optimize()

#         if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
#             plt.scatter(*zip(*points))
#             plt.figure()
#             # LP obj value
#             lp_obj = 0
#             for i, v in list(num_select.items()):
#                 if v.X == 0:
#                     continue
#                 center = (group_centroid[i, 0].X, group_centroid[i, 1].X)
#                 for j in index:
#                     if x[j, i].X > 0.5:
#                         plt.scatter(*points[j], color=colors[i])
#                         lp_obj += np.linalg.norm(np.array(points[j]) - np.array(center))
#                 plt.scatter(*center, color=colors[i], marker="x")
#             # kmean obj value
#             km_obj = 0
#             for i, point in enumerate(points):
#                 km_obj += np.linalg.norm(
#                     np.array(point) - np.array(km.cluster_centers_[km.labels_[i]])
#                 )
#             plot_images(plt.gcf(), 500)
#             return lp_obj, km_obj


def draw_boxplot(nested_list, sample_sizes, category_names):
    # Original nested list
    # nested_list = [[(1, 4), (2, 5), (3, 6)], [(1, 4), (2, 5), (3, 6)]]
    # Convert the nested list to a flat list with associated categories
    data = []
    for i, category in enumerate(nested_list):
        category = list(zip(*category))
        for j, subcategory in enumerate(category):
            for value in subcategory:
                data.append(
                    {
                        "Sample Sizes": f"{sample_sizes[i]}",  # Assign category names
                        "Subcategory": f"{category_names[j]}",  # Assign subcategory names
                        "Value": value,
                    }
                )
    df = pd.DataFrame(data)
    sns.boxplot(x="Sample Sizes", y="Value", hue="Subcategory", data=df)
    plot_images(plt.gcf(), 500)


def single_test(num_points):
    np.random.seed(0)
    points = [(np.random.random() * 100, np.random.random() * 300) for _ in range(num_points)]
    points = np.array(points)
    km = lp_plot(points, N=4, method=1)
    evaluate(points, km)
    exit()


statistics = []
sample_sizes = [5, 10, 20, 50, 100, 150, 200]
sample_sizes = [200]
# plot_images.disable = True
for num_points in sample_sizes:
    statistic = []
    K = num_points // 4 + 1  # Number
    colors = np.random.rand(K, 3)
    for seed in range(1):
        np.random.seed(seed)
        points = [(np.random.random() * 100, np.random.random() * 200) for _ in range(num_points)]
        points.sort(key=lambda x: x[0] + x[1])
        points = np.array(points)
        min_km_value = 1e9
        min_km = None
        for _ in range(5):
            km = kmean_alg(method=1, points=points, N=4)
            value = evaluate(points, km)
            if value < min_km_value:
                min_km_value = value
                min_km = km
        draw(min_km, colors)
        print(np.bincount(min_km.labels_))
        exit()
        km = lp_plot(points, N=4, method=1, km=min_km)
        value1 = evaluate(points, km)
        draw(km, colors)
        km = lp_plot(points, N=4, method=2, km=min_km)
        value2 = evaluate(points, km)
        draw(km, colors)
        statistic.append((value1, value2))
    statistics.append(statistic)

plot_images.disable = False
draw_boxplot(statistics, sample_sizes, ["Method 1", "Method 2"])
exit()
plot_images.disable = True
nested_list = []
sample_sizes = [5, 10, 20]
category_names = ["LP", "KMeans"]
for k in sample_sizes:
    print("Sample size:", k)
    data = []
    for i in range(5):
        result = test(i, k)
        data.append(result)
    nested_list.append(data)

plot_images.disable = False
