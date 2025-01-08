import math
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import gurobipy as gp
import matplotlib.pyplot as plt
import scipy
from gurobipy import GRB, Model, quicksum
from sklearn.cluster import KMeans

sys.path.append("hello_world/src")

import cv2
import numpy as np
from plot import *
from tqdm import tqdm
from utility_image_wo_torch import *


def hello_world():
    print("Hello World!")


def draw_layout(
    display_in_shell,
    file_name,
    die_size,
    bin_width,
    bin_height,
    placement_rows,
    flip_flops,
    gates,
    io_pins,
    extra_visual_elements,
):
    BORDER_COLOR = (46, 117, 181)
    PLACEROW_COLOR = (0, 111, 162)
    FLIPFLOP_COLOR = (165, 226, 206)
    FLIPFLOP_WALKED_COLOR = (255, 255, 0)
    FLIPFLOP_OUTLINE_COLOR = (84, 90, 88)
    GATE_COLOR = (237, 125, 49)
    GATE_WALKED_COLOR = (0, 0, 255)
    IO_PIN_OUTLINE_COLOR = (54, 151, 217)
    img_width = die_size.x_upper_right
    img_height = die_size.y_upper_right
    max_length = 8000
    ratio = max_length / max(img_width, img_height)
    img_width, img_height = int(img_width * ratio), int(img_height * ratio)
    img = np.full((img_height, img_width, 3), 255, np.uint8)
    border_width = int(max_length * 0.02)
    line_width = int(max_length * 0.003)
    dash_length = int(max_length * 0.01)

    # Draw shaded bins
    for i in range(0, math.ceil(die_size.x_upper_right / bin_width)):
        for j in range(0, math.ceil(die_size.y_upper_right / bin_height)):
            if i % 2 == 0:
                if j % 2 == 1:
                    continue
            elif j % 2 == 0:
                continue
            start = (i * bin_width * ratio, j * bin_height * ratio)
            end = ((i + 1) * bin_width * ratio, (j + 1) * bin_height * ratio)
            start = np.int32(start)
            end = np.int32(end)
            x, y, w, h = start[0], start[1], end[0] - start[0], end[1] - start[1]
            sub_img = img[y : y + h, x : x + w]
            white_rect = np.full(sub_img.shape, (120, 120, 120), dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 0)
            img[y : y + h, x : x + w] = res
    # Draw the placement rows
    dash_line_width = int(abs(placement_rows[0].y - placement_rows[1].y) * ratio / 2.5)
    dash_length = int(placement_rows[0].width * ratio) * 3
    bin_site_ratio = bin_height // placement_rows[0].height
    for row_idx, placement_row in enumerate(placement_rows):
        x, y = placement_row.x, placement_row.y
        w = bin_width
        h = placement_row.height
        x, y, w, h = int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)
        dashed_line(
            img,
            (x, y),
            (x + w, y),
            PLACEROW_COLOR,
            dash_line_width,
            dash_length=dash_length,
            gap_length=dash_length,
        )
        if row_idx > bin_site_ratio:
            break
        # for i in range(1, placement_row.num_cols):
        #     x = placement_row.x + i * placement_row.width
        #     x = int(x * ratio)
        #     dashed_line(
        #         img,
        #         (x, y),
        #         (x, y + h),
        #         PLACEROW_COLOR,
        #         line_width,
        #         dash_length=dash_length,
        #         gap_length=int(dash_length / 1.5),
        #     )

    # Draw the flip-flops
    highlighted_cell = []
    for ff in flip_flops:
        x, y = ff.x, ff.y
        w = ff.width
        h = ff.height
        x, y = int(x * ratio), int(y * ratio)
        w, h = int(w * ratio), int(h * ratio)
        half_border_width = max(max(w, h) // 70, 1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            FLIPFLOP_COLOR if not ff.walked else FLIPFLOP_WALKED_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), FLIPFLOP_OUTLINE_COLOR, half_border_width * 2)
        if ff.highlighted:
            highlighted_cell.append((x, y, w, h, half_border_width))

    # Draw the gates
    for gate in gates:
        x, y = gate.x, gate.y
        w = gate.width
        h = gate.height
        x, y = int(x * ratio), int(y * ratio)
        w, h = int(w * ratio), int(h * ratio)
        half_border_width = max(max(w, h) // 70, 1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            GATE_COLOR if not gate.walked else GATE_WALKED_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), FLIPFLOP_OUTLINE_COLOR, half_border_width * 2)
        if gate.highlighted:
            highlighted_cell.append((x, y, w, h, half_border_width))

    # Draw the io pins
    io_pin_width = max(min(min(gate.width, gate.height) for gate in gates) // 3, 1)
    for io in io_pins:
        x, y = io.x, io.y
        w = h = io_pin_width
        w, h = int(w * ratio), int(h * ratio)
        x, y = int((x) * ratio), int(y * ratio)
        if img_width - x < w:
            x = img_width - w
        half_border_width = int(max(w, h) * 0.1)
        cv2.rectangle(
            img,
            (x + half_border_width, y + half_border_width),
            (x + w - half_border_width, y + h - half_border_width),
            BORDER_COLOR,
            -1,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), IO_PIN_OUTLINE_COLOR, half_border_width * 2)

    # Highlight the selected cells
    for cell in highlighted_cell:
        x, y, w, h, half_border_width = cell
        size = 5
        cv2.rectangle(
            img,
            (x - size * w, y - size * h),
            (x + (size + 1) * w, y + (size + 1) * h),
            (0, 0, 0),
            half_border_width * 15,
        )

    for extra in extra_visual_elements:
        rect = (
            (extra[0] * ratio, extra[1] * ratio),
            (extra[2] * ratio, extra[3] * ratio),
            extra[4],
        )
        box = cv2.boxPoints(rect)  # Get the four corners
        box = np.int0(box)  # Convert to integer

        # Draw the rotated rectangle
        cv2.polylines(img, [box], isClosed=True, color=(0, 0, 0), thickness=max_length // 500)
    img = cv2.flip(img, 0)

    # Add a border around the image
    img = cv2.copyMakeBorder(
        img,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=BORDER_COLOR,
    )
    img = cv2.copyMakeBorder(
        img,
        top=border_width * 2,
        bottom=border_width * 2,
        left=border_width * 2,
        right=border_width * 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if display_in_shell:
        plot_images(img, 600)
    cv2.imwrite(file_name, img)
    print(f"Image saved to {file_name}")
    P = PlotlyUtility(file_name, margin=0, showaxis=False)
    P.add_image(img)
    P.show(save=False)


@dataclass
class VisualizeOptions:
    pin_text: bool = True
    pin_marker: bool = True
    line: bool = True
    cell_text: bool = True
    io_text: bool = True
    placement_row: bool = False


def visualize(
    file_name, die_size, bin_width, bin_height, placement_rows, flip_flops, gates, io_pins, nets
):
    options = VisualizeOptions(
        line=True,
        cell_text=True,
        io_text=True,
        placement_row=False,
    )

    P = PlotlyUtility(file_name=file_name if file_name else "output.html", margin=30)
    P.add_rectangle(
        BoxContainer(
            die_size.x_upper_right - die_size.x_lower_left,
            die_size.y_upper_right - die_size.y_lower_left,
            offset=(die_size.x_lower_left, die_size.y_lower_left),
        ).box,
        color_id="black",
        fill=False,
        group="die",
    )

    for i in range(0, math.ceil(die_size.x_upper_right / bin_width)):
        for j in range(0, math.ceil(die_size.y_upper_right / bin_height)):
            if i % 2 == 0:
                if j % 2 == 1:
                    continue
            else:
                if j % 2 == 0:
                    continue
            P.add_rectangle(
                BoxContainer(
                    bin_width,
                    bin_height,
                    offset=(i * bin_width, j * bin_height),
                ).box,
                color_id="rgba(44, 44, 160, 0.3)",
                line_color="rgba(0,0,0,0)",
                fill=True,
                group="bin",
            )
    # if options.placement_row:
    #     for row in setting.placement_rows:
    #         P.add_line(
    #             (row.x, row.y),
    #             (row.x + row.width * row.num_cols, row.y),
    #             group="row",
    #             line_width=1,
    #             line_color="black",
    #             dash=False,
    #         )
    #         for i in range(row.num_cols):
    #             P.add_line(
    #                 (row.x + i * row.width, row.y),
    #                 (row.x + i * row.width, row.y + row.height),
    #                 group="row",
    #                 line_width=1,
    #                 line_color="black",
    #                 dash=False,
    #             )

    #         # print(row)
    #         # exit()
    #         # for i in range(int(row.num_cols)):
    #         #     P.add_line(
    #         #         (row.x + i * row.width, row.y),
    #         #         (row.x + i * row.width, row.y + row.height),
    #         #         group="row",
    #         #         line_width=1,
    #         #         line_color="black",
    #         #         dash=False,
    #         #     )
    #         # P.add_rectangle(
    #         #     BoxContainer(row.width, row.height, offset=(row.x + i * row.width, row.y)).box,
    #         #     color_id="black",
    #         #     fill=False,
    #         #     group=1,
    #         #     dash=True,
    #         #     line_width=1,
    #         # )
    if len(flip_flops) + len(gates) <= 15:
        options.pin_marker = True
        options.pin_text = True
    else:
        options.pin_marker = False
        options.pin_text = False

    for input in io_pins:
        P.add_rectangle(
            BoxContainer(2, 0.8, offset=(input.x, input.y), centroid="c").box,
            color_id="red",
            group="input",
            text_position="top centerx",
            fill_color="red",
            text=input.name if options.io_text else None,
            show_marker=False,
        )

    for flip_flop in tqdm(flip_flops):
        inst_box = BoxContainer(
            flip_flop.width, flip_flop.height, offset=(flip_flop.x, flip_flop.y)
        )
        P.add_rectangle(
            inst_box.box,
            color_id="rgba(100,183,105,1)",
            group="ff",
            line_color="black",
            bold=True,
            text=flip_flop.name if options.cell_text else None,
            label=flip_flop.name,
            text_position="centerxy",
            show_marker=False,
        )
        if options.pin_marker:
            for pin in flip_flop.pins:
                pin_box = BoxContainer(0, offset=(pin.x, pin.y))
                P.add_rectangle(
                    pin_box.box,
                    group="ffpin",
                    text=pin.name if options.pin_text else None,
                    text_location=(
                        "middle right" if pin_box.left < inst_box.centerx else "middle left"
                    ),
                    marker_size=8,
                    marker_color="rgb(255, 200, 23)",
                )
    for gate in gates:
        inst_box = BoxContainer(gate.width, gate.height, offset=(gate.x, gate.y))
        P.add_rectangle(
            inst_box.box,
            color_id="rgba(239,138,55,1)",
            group="gate",
            line_color="black",
            bold=True,
            text=gate.name if options.cell_text else None,
            # label=inst.lib.name,
            text_position="centerxy",
            show_marker=False,
        )
        if options.pin_marker:
            for pin in gate.pins:
                pin_box = BoxContainer(0, offset=(pin.x, pin.y))
                P.add_rectangle(
                    pin_box.box,
                    group="gatepin",
                    text=pin.name if options.pin_text else None,
                    text_location=(
                        "middle right" if pin_box.left < inst_box.centerx else "middle left"
                    ),
                    text_color="black",
                    marker_size=8,
                    marker_color="rgb(255, 200, 23)",
                )

    if options.line:
        for net in nets:
            if net.is_clk:
                continue
            starting_pin = net.pins[0]
            for pin in net.pins[1:]:
                # if pin.name.lower() == "clk" or starting_pin.name.lower() == "clk":
                #     continue
                # if pin.inst.name == starting_pin.inst.name:
                #     continue
                P.add_line(
                    start=(starting_pin.x, starting_pin.y),
                    end=(pin.x, pin.y),
                    line_width=2,
                    line_color="black",
                    group="net",
                    # text=net.metadata,
                )
    P.show(save=True)


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


def evaluate(points, centers, labels, output=False):
    km_obj = 0
    for i, point in enumerate(points):
        km_obj += np.linalg.norm(np.array(point) - np.array(centers[labels[i]]))
    if output:
        print("objective value:", km_obj)
    return km_obj


def plot_points_with_centers(points, centers, labels, colors):
    plt.figure()
    for i, point in enumerate(points):
        plt.scatter(*point, color=colors[labels[i]])
    plt.scatter(*zip(*centers), color=colors, marker="x")
    plot_images(plt.gcf(), 500)


def lp_plot(points, N, method, labels):
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
            # model.Params.TimeLimit = 60
            model.Params.WorkLimit = 120
            model.Params.Threads = min(24, os.cpu_count())
            # model.Params.ScaleFlag = 1
            # model.Params.OutputFlag = 0
            # model.Params.MIPFocus = 0
            # model.Params.LogToConsole = 0
            # model.Params.SolutionLimit = 1
            # model.Params.MIPGap = 0.5
            x = model.addVars(index, cindex, vtype=GRB.BINARY, name="x")
            num_select = model.addVars(cindex, ub=N, vtype=GRB.INTEGER)
            # selected = model.addVars(cindex, vtype=GRB.BINARY)
            gap_M = np.linalg.norm(np.max(points, axis=0)) * 10
            # non_empty_col = model.addVars(cindex, vtype=GRB.BINARY)
            group_centroid = model.addVars(cindex, [0, 1], ub=gap_M, vtype=GRB.CONTINUOUS)
            group_centroid_gap = model.addVars(
                index, cindex, [0, 1], lb=-GRB.INFINITY, ub=gap_M, vtype=GRB.CONTINUOUS
            )
            group_centroid_gap_positive = model.addVars(
                index, cindex, [0, 1], lb=-GRB.INFINITY, ub=gap_M, vtype=GRB.CONTINUOUS
            )
            group_centroid_gap_abs = model.addVars(
                index, cindex, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY
            )
            # min_val = model.addVars(cindex, vtype=GRB.CONTINUOUS)
            # warm start
            if labels is not None:
                for i in index:
                    for j in cindex:
                        x[i, j].start = 0
                    label = labels[i]
                    x[i, label].start = 1

            for i in index:
                model.addConstr(quicksum(x[i, j] for j in cindex) == 1)
            for j in cindex:
                model.addConstr(quicksum(x[i, j] for i in index) <= N)
                # model.addConstr(min_val[j] == gp.min_(x[i, j] for i in index))
            for j in cindex:
                model.addConstr(num_select[j] == quicksum(x[i, j] for i in index))
            for j in cindex:
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
                            (group_centroid_gap[i, j, 0] == (group_centroid[j, 0] - points[i][0]))
                        )
                        model.addConstr(
                            (group_centroid_gap[i, j, 1] == (group_centroid[j, 1] - points[i][1]))
                        )
                        model.addConstr(
                            group_centroid_gap_positive[i, j, 0]
                            == gp.abs_(group_centroid_gap[i, j, 0])
                        )
                        model.addConstr(
                            group_centroid_gap_positive[i, j, 1]
                            == gp.abs_(group_centroid_gap[i, j, 1])
                        )
                        model.addConstr(
                            group_centroid_gap_abs[i, j]
                            == (
                                group_centroid_gap_positive[i, j, 0]
                                + group_centroid_gap_positive[i, j, 1]
                            )
                            * x[i, j]
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
            print(e)

        if (
            model.status == GRB.OPTIMAL
            or model.status == GRB.TIME_LIMIT
            or model.status == GRB.WORK_LIMIT
            or model.SolCount > 0
        ):
            plt.scatter(*zip(*points))
            plt.figure()
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


def plot_kmeans_output(pyo3_kmeans_result):
    points = np.reshape(pyo3_kmeans_result.points, (-1, 2))
    centers = np.reshape(pyo3_kmeans_result.cluster_centers, (-1, 2))
    labels = pyo3_kmeans_result.labels
    colors = np.random.rand(centers.shape[0], 3)
    min_km_value = 1e9
    min_km = None
    for _ in range(5):
        km = kmean_alg(method=2, points=points, N=4)
        value = evaluate(points, km.cluster_centers_, km.labels_, False)
        if value < min_km_value:
            min_km_value = value
            min_km = km
    print("py")
    plot_points_with_centers(points, min_km.cluster_centers_, min_km.labels_, colors)
    print("min_km_value:", min_km_value)
    print("rust")
    plot_points_with_centers(points, centers, labels, colors)
    evaluate(points, centers, labels, True)
    # lp_plot(points, 4, method=2, labels=labels)


def single_test(num_points):
    np.random.seed(0)
    points = [(np.random.random() * 100, np.random.random() * 300) for _ in range(num_points)]
    points = np.array(points)
    km = kmean_alg(method=2, points=points, N=4)
    evaluate(points, km.cluster_centers_, km.labels_, True)
    print(np.bincount(km.labels_))
    bad_group = np.where(np.bincount(km.labels_) < 4)[0]
    print(bad_group)


def plot_binary_image(arr, aspect_ratio=1, title="", grid=False):
    img = np.array(arr)
    img = np.flip(img, 0)
    plt.imshow(
        1 - img,
        cmap="gray" if img.sum() > 0 else "gray_r",
        extent=(0, img.shape[1], 0, img.shape[0]),
    )
    plt.gca().set_aspect(aspect_ratio)
    if grid:
        plt.gca().set_xticks(range(img.shape[1]))
        plt.gca().set_yticks(range(img.shape[0]))
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.grid()
    if title:
        plt.title(title)
    plt.gca().figure.tight_layout()
    plot_images(plt.gcf(), 500)
    plt.close()


def solve_tiling_problem(grid_size, tiles, tiles_weight, tile_limits, spatial_occupancy):
    import gurobipy as gp
    from gurobipy import GRB

    # Define grid and tile sizes
    N, M = grid_size  # Grid size (width, height)
    # tiles = [(1, 2), (2, 1), (2, 2)]  # Tile types (width, height)
    # tile_limits = [20, 25, 2]  # Limits for each tile type (max tiles)
    # spatial_occupancy = np.transpose(spatial_occupancy)
    # print(spatial_occupancy.shape)
    # exit()
    tiles_area = [w * h for w, h in tiles]

    # Create model
    model = gp.Model("RectangularTiling")
    model.Params.Threads = min(24, os.cpu_count())
    # Decision variables
    x = model.addVars(len(tiles), N, M, vtype=GRB.BINARY, name="x")  # Tile placement
    y = model.addVars(N, M, vtype=GRB.BINARY, name="y")  # Cell coverage
    y_weight = model.addVars(N, M, vtype=GRB.CONTINUOUS)

    # Spatial occupancy constraints
    # if len(spatial_occupancy) > 0:
    #     for i in range(N):
    #         for j in range(M):
    #             if spatial_occupancy[i][j]:
    #                 for k in range(len(tiles)):
    #                     model.addConstr(x[k, i, j] == 0)

    # Tile placement constraints
    for k, (tile_w, tile_h) in enumerate(tiles):
        for i in range(N - tile_w + 1, N):
            model.addConstr(gp.quicksum(x[k, i, j] for j in range(M)) == 0)
        for j in range(M - tile_h + 1, M):
            model.addConstr(gp.quicksum(x[k, i, j] for i in range(N)) == 0)

    # Coverage constraints
    for i in range(N):
        for j in range(M):
            # A cell (i, j) is covered if it's within the bounds of any tile placement
            model.addConstr(
                y[i, j]
                == gp.quicksum(
                    x[k, r, c]
                    for k, (tile_w, tile_h) in enumerate(tiles)
                    for r in range(max(0, i - tile_w + 1), i + 1)
                    for c in range(max(0, j - tile_h + 1), j + 1)
                    if r + tile_w <= N and c + tile_h <= M
                )
                + spatial_occupancy[i][j],
                name=f"cover_{i}_{j}",
            )
            model.addConstr(
                y_weight[i, j]
                == gp.quicksum(
                    x[k, r, c] / tiles_area[k] * tiles_weight[k]
                    for k, (tile_w, tile_h) in enumerate(tiles)
                    for r in range(max(0, i - tile_w + 1), i + 1)
                    for c in range(max(0, j - tile_h + 1), j + 1)
                    if r + tile_w <= N and c + tile_h <= M
                ),
                name=f"cover_{i}_{j}",
            )

    # Non-overlapping constraints: Each cell can be covered by at most one tile
    for i in range(N):
        for j in range(M):
            model.addConstr(y[i, j] <= 1, name=f"no_overlap_{i}_{j}")

    # Tile count limits
    if tile_limits != 0:
        for k, (tile_w, tile_h) in enumerate(tiles):
            model.addConstr(
                gp.quicksum(x[k, i, j] for i in range(N) for j in range(M)) == tile_limits[k],
                name=f"tile_limit_{k}",
            )

    # Objective: Maximize total coverage
    # model.setObjective(gp.quicksum(y[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE)
    model.setObjective(
        gp.quicksum(y_weight[i, j] for i in range(N) for j in range(M)), GRB.MAXIMIZE
    )

    # Solve the model
    model.optimize()

    # Print solution
    if model.status == GRB.OPTIMAL:
        # print("Optimal solution found!")
        # for k, (tile_w, tile_h) in enumerate(tiles):
        #     print(f"Tile type {k} ({tile_h}x{tile_w}):")
        #     for i in range(N):
        #         for j in range(M):
        #             if x[k, i, j].x > 0.5:
        #                 print(f"  Placed at ({i}, {j})")

        # draw layout using matplotlib
        layout = np.zeros((N, M))
        for k, (tile_h, tile_w) in enumerate(tiles):
            for i in range(N):
                for j in range(M):
                    if x[k, i, j].x > 0.5:
                        layout[i : i + tile_h, j : j + tile_w] = k + 1

        if len(spatial_occupancy) > 0:
            for i in range(N):
                for j in range(M):
                    if spatial_occupancy[i][j]:
                        for k in range(len(tiles)):
                            layout[i, j] = len(tiles) + 1
        plot_binary_image(layout, aspect_ratio=1, title="Tile placements", grid=True)
        # layout = np.zeros((N, M))
        # for i in range(N):
        #     for j in range(M):
        #         layout[i, j] = y_weight[i, j].x
        # plot_binary_image(layout, aspect_ratio=1, title="Tile placements", grid=True)
        capcaity = np.zeros(len(tiles), dtype=int)
        for k, (tile_h, tile_w) in enumerate(tiles):
            for i in range(N):
                for j in range(M):
                    if x[k, i, j].x > 0.5:
                        capcaity[k] += 1
            print(f"Tile type {k} ({tile_h}x{tile_w}): {int(capcaity[k])}")

        # print(list(map(lambda v: v.x, y.values())))
    else:
        print("No solution found.")


if __name__ == "__main__":
    # single_test(102)
    k = 30
    map = np.zeros((k, k // 2))
    map[0, 0] = 1
    map[1, 1] = 1
    map[2, 0] = 1
    solve_tiling_problem((k, k // 2), [(2, 2), (2, 1)], [2.4, 1], 0, map)
    pass
