import math
import sys
from collections import defaultdict
from pprint import pprint

import numpy as np
import rustlib
import shapely
from llist import dllist, sllist
from shapely.geometry import Polygon

from faketime_utl import ensure_time
from mbffg import MBFFG, VisualizeOptions
from utility import *

# ensure_time()
if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
else:
    # input_path = "temp.out"
    # input_path = "cases/new_c2.txt"
    # input_path = "cases/new_c4.txt"
    output_path = "cases/output/output.txt"
    input_path = "cases/new_c5.txt"
    input_path = "cases/new_c1.txt"
    input_path = "cases/new_c3.txt"
    input_path = "cases/testcase1.txt"
    input_path = "cases/v2.txt"
    input_path = "cases/sample.txt"
    input_path = "cases/testcase1_0614.txt"
    input_path = "cases/testcase0.txt"

options = VisualizeOptions(
    line=True,
    cell_text=True,
    io_text=False,
    placement_row=True,
)
mbffg = MBFFG(input_path)
mbffg.cvdraw()
if mbffg.G.size < 1000:
    mbffg.transfer_graph_to_setting(options=options)

ori_score = mbffg.scoring()


# print(ori_score)
# exit()
def calculate_potential_space(mbffg: MBFFG):
    row_coordinates = [
        x.get_rows() for x in sorted(mbffg.setting.placement_rows, key=lambda x: x.y)
    ]
    obstacles = [x.bbox_corner for x in mbffg.get_gates()]
    optimal_library_segments, library_sizes = mbffg.get_selected_library()
    grid_sizes = [(x.width, x.height) for x in optimal_library_segments.values()]
    placement_resource_map = rustlib.placement_resource(row_coordinates, obstacles, grid_sizes)
    placement_resource_map = np.array(placement_resource_map)
    # bool_map[0, 0, 0] = False
    candidate_indices = [
        [npindex(map[i], True)[0] for i in range(len(grid_sizes))] for map in placement_resource_map
    ]
    num_rows = len(row_coordinates)
    potential_space = [0] * len(grid_sizes)
    for current_idx in range(len(grid_sizes)):
        for i in range(num_rows):
            start_idx = candidate_indices[i][current_idx]
            while start_idx is not None:
                if placement_resource_map[i][current_idx][start_idx] == False:
                    following_index = npindex(
                        placement_resource_map[i][current_idx], True, start_idx + 1
                    )
                    if following_index is None:
                        break
                    else:
                        start_idx = start_idx + 1 + following_index[0]
                    continue
                potential_space[current_idx] += 1
                if grid_sizes[current_idx][0] > mbffg.setting.placement_rows[i].width:
                    w = math.ceil(
                        grid_sizes[current_idx][0] / mbffg.setting.placement_rows[i].width
                    )
                    assert placement_resource_map[i][current_idx][start_idx]
                    effect_range = placement_resource_map[i][:, start_idx : start_idx + w]
                    effect_range[:] = False
                from_value = (
                    mbffg.setting.placement_rows[i].x
                    + start_idx * mbffg.setting.placement_rows[i].width
                )
                to_value = from_value + grid_sizes[current_idx][0]
                # h = math.ceil(candidates[current_idx][1] / mbffg.setting.placement_rows[i].height)
                for j in range(i + 1, num_rows):
                    if (
                        mbffg.setting.placement_rows[i].y + grid_sizes[current_idx][1] - 1e-4
                        >= mbffg.setting.placement_rows[j].y
                    ):
                        start = math.floor(
                            (from_value - mbffg.setting.placement_rows[j].x)
                            / mbffg.setting.placement_rows[j].width
                        )
                        w = math.ceil(
                            (to_value - mbffg.setting.placement_rows[j].x)
                            / mbffg.setting.placement_rows[j].width
                        )
                        effect_range = placement_resource_map[j][:, start : start + w]
                        effect_range[:] = False
    return potential_space


potential_space = calculate_potential_space(mbffg)


def potential_space_cluster(potential_space):
    optimal_library_segments, library_sizes = mbffg.get_selected_library()
    ffs = set([x.name for x in mbffg.get_ffs()])
    while ffs:
        ff = next(iter(ffs))
        subg = mbffg.G_clk.neighbors(ff) + [ff]
        size = len(subg)
        if size == 1:
            ffs -= set(subg)
            continue
        # library_sizes.sort(key=lambda x: abs(x - size))
        lib_idx = index(
            list(enumerate(library_sizes)), lambda x: x[1] <= size and potential_space[x[0]] > 0
        )
        # if lib_idx is None:
        #     print(lib_idx)
        #     print(potential_space)
        #     print(size)
        potential_space[lib_idx] -= 1
        size = library_sizes[lib_idx]
        g = subg[:size]
        mbffg.merge_ff(",".join(g), optimal_library_segments[size].name)
        # ff has no neighbors
        mbffg.G_clk.remove_nodes(g)
        ffs -= set(g)


potential_space_cluster(potential_space)


# print(mbffg.G.edges())
# exit()
# for name in mbffg.G.node_names():
#     print(name)
#     print(mbffg.G.neighbors(name))
#     print(mbffg.G.get_ancestor_until(name, 1))
# exit()
# ori_score = mbffg.scoring()
def clustering():
    def slack_region(pos, slack):
        x, y = pos
        return Polygon([(x, y + slack), (x + slack, y), (x, y - slack), (x - slack, y)])

    def rotate_45(intersection_coord):
        intersection_coord[:, 0], intersection_coord[:, 1] = (
            intersection_coord[:, 0] + intersection_coord[:, 1],
            intersection_coord[:, 1] - intersection_coord[:, 0],
        )

    interval_graph_x = []
    interval_graph_y = []
    flip_flops = mbffg.get_ffs()
    library_sorted = sorted(
        mbffg.get_library().values(),
        key=lambda x: ((x.power * mbffg.setting.beta + x.area * mbffg.setting.gamma) / x.bits,),
    )
    library_seg_best = {}
    for lib in library_sorted:
        if lib.bits not in library_seg_best:
            library_seg_best[lib.bits] = lib
    # library_sorted = sorted(library_seg_best.items())
    # filename = "slackregion.html" if len(flip_flops) < 1000 else "slackregion.png"
    # Ptest = PlotlyUtility(file_name=filename, margin=30)
    acceptable_cost = (
        (library_sorted[0].power * mbffg.setting.beta)
        + (library_sorted[0].area * mbffg.setting.gamma)
    ) / (library_sorted[0].bits * mbffg.setting.alpha)

    for ffidx, ff in enumerate(flip_flops):
        dd = (ff.lib.power * mbffg.setting.beta + ff.lib.area * mbffg.setting.gamma) / (
            ff.lib.bits * mbffg.setting.alpha
        ) - acceptable_cost
        dd = (mbffg.get_pin(ff.dpins[0]).slack + dd) / mbffg.setting.displacement_delay / 2
        if dd < 0:
            continue
        ff.metadata.moveable_area = slack_region(
            ff.center,
            dd,
        )
        intersection_coord = shapely.get_coordinates(ff.metadata.moveable_area)
        # Ptest.add_rectangle(
        #     BoxContainer(0, offset=ff.center).box,
        #     group="ffpin",
        #     marker_size=10,
        #     marker_color="rgb(255, 200, 23)",
        #     text="nil",
        # )
        # Ptest.add_rectangle(intersection_coord, fill=False, group=0)
        rotate_45(intersection_coord)

        intersection_coord = intersection_coord[:-1]
        regionx, regiony = intersection_coord[:, 0], intersection_coord[:, 1]
        xstart, xend = regionx.min(), regionx.max()
        ystart, yend = regiony.min(), regiony.max()
        if xstart >= xend:
            xend += (xstart - xend) + 1e-8
        if ystart >= yend:
            yend += (ystart - yend) + 1e-8
        interval_graph_x.append((xstart, ffidx, True))
        interval_graph_x.append((xend, ffidx, False))
        interval_graph_y.append((ystart, ffidx, True))
        interval_graph_y.append((yend, ffidx, False))

    # Ptest.show(save=True)
    interval_graph_x.sort(key=lambda x: (x[0], x[2]))
    interval_graph_x = sllist(interval_graph_x)
    interval_graph_y.sort(key=lambda x: (x[0], x[2]))
    interval_graph_y = sllist(interval_graph_y)

    interval_graph_x_inv = defaultdict(list)
    interval_graph_y_inv = defaultdict(list)
    current_node = interval_graph_x.first
    i = 0
    while current_node:
        interval_graph_x_inv[current_node.value[1]].append(current_node)
        current_node = current_node.next
        i += 1
    current_node = interval_graph_y.first
    while current_node:
        interval_graph_y_inv[current_node.value[1]].append(current_node)
        current_node = current_node.next
    K = []

    def run(required_endpoint):
        related_ff = set()
        related_ff_ls = dllist()
        related_ff_ls_inv = defaultdict(list)
        current_node = interval_graph_x.first
        any_decision = False
        while current_node:
            ff_name, ff_is_start = current_node.value[1], current_node.value[2]
            related_ff_ls.append(current_node)
            related_ff_ls_inv[current_node.value[1]].append(related_ff_ls.last)

            # print("--", flip_flops[ff_name].name, related_ff_ls.size)
            # print(current_node, current_node.next)
            # found decision point
            if (
                related_ff_ls.size > 1
                and not related_ff_ls.last.value.value[2]
                and related_ff_ls.last.prev.value.value[2]
                # and related_ff_ls.last.value.value[1] != related_ff_ls.last.prev.value.value[1]
            ):
                y_interval_start = False
                related_ff_y = set()
                related_ff_y_ls = dllist()
                max_clique = []
                current_node_y = interval_graph_y.first
                # print(f"decision point x {ff_name}, len {len(interval_graph_x)}")
                while current_node_y:
                    ff_name_y, ff_is_start_y = current_node_y.value[1], current_node_y.value[2]
                    if ff_name_y in related_ff:
                        related_ff_y_ls.append(current_node_y)
                        if (
                            y_interval_start
                            and ff_name_y == ff_name
                            and not ff_is_start_y
                            and required_endpoint
                        ):
                            break
                        if (
                            y_interval_start
                            and not related_ff_y_ls.last.value.value[2]
                            and related_ff_y_ls.last.prev.value.value[2]
                        ):
                            # print(f"decision point y {ff_name_y}")
                            max_clique = related_ff_y.intersection(related_ff)
                            max_clique_with_same_clk_net = set()
                            decision_clk_net = mbffg.get_pin(flip_flops[ff_name].clkpin).net_name
                            for c in max_clique:
                                if mbffg.get_pin(flip_flops[c].clkpin).net_name == decision_clk_net:
                                    max_clique_with_same_clk_net.add(c)
                            max_clique = max_clique_with_same_clk_net
                            break
                        if (
                            y_interval_start
                            and ff_name_y == ff_name
                            and not ff_is_start_y
                            and not required_endpoint
                        ):
                            break
                        if ff_name_y == ff_name and ff_is_start_y:
                            y_interval_start = True

                        if ff_is_start_y:
                            related_ff_y.add(ff_name_y)
                            # print("+", ff_name_y)
                        else:
                            related_ff_y.remove(ff_name_y)
                            # print("-", ff_name_y)
                    current_node_y = current_node_y.next
                if len(max_clique) > 0:
                    # find appropriate library
                    B = 0
                    B2 = 0
                    clique_size = sum([flip_flops[c].bits for c in max_clique])
                    for lib in library_sorted:
                        if lib.bits <= clique_size:
                            B = lib.bits
                            B2 = B
                            Btmp = lib.name
                            # print(f"choose lib {lib.bits}")
                            break

                    max_clique.remove(ff_name)
                    decision_point_ff = flip_flops[ff_name]
                    B -= decision_point_ff.bits
                    # print("remain size", B)
                    k = [ff_name]

                    max_clique = list(max_clique)
                    max_clique.sort(
                        key=lambda x: flip_flops[x].bits,
                        reverse=True,
                    )
                    # print(flip_flops[0])
                    # exit()
                    for c in max_clique:
                        bit = flip_flops[c].bits
                        t = B - bit
                        if t >= 0:
                            # print("select", c, bit)
                            B = t
                            k.append(c)
                        if t == 0:
                            break
                    if B != 0:
                        print("error")
                        # for c in max_clique:
                        # bit = flip_flop_list[flip_flop_names[c]].library
                        # print(c, bit)
                        current_node = current_node.next
                    elif B2 == 1:
                        current_node = current_node.next
                    else:
                        any_decision = True
                        K.append({"bit": Btmp, "ff": k})
                        # print("remove", k)
                        tmp = current_node.next
                        while tmp and tmp.value[1] in k:
                            tmp = tmp.next
                        current_node = tmp
                        # print("current_node move to", current_node)
                        for kele in k:
                            for node in interval_graph_x_inv[kele]:
                                interval_graph_x.remove(node)
                            for node in related_ff_ls_inv[kele]:
                                related_ff_ls.remove(node)
                            for node in interval_graph_y_inv[kele]:
                                interval_graph_y.remove(node)
                            if kele != ff_name:
                                related_ff.remove(kele)
                else:
                    # print("no clique found")
                    current_node = current_node.next

            else:
                current_node = current_node.next
            # print("related ff:", related_ff)
            if ff_is_start:
                related_ff.add(ff_name)
                # print("+", ff_name)
            else:
                related_ff.remove(ff_name)
                # print("-", ff_name)
        return any_decision

    # with HiddenPrints():
    run(False)
    for k in K:
        mbffg.merge_ff(",".join([flip_flops[x].name for x in k["ff"]]), k["bit"])
    print(f"merge {len(K)} flip-flops")


def clustering_random():
    library_seg_best, lib_keys = mbffg.get_selected_library()
    ffs = set([x.name for x in mbffg.get_ffs()])
    while ffs:
        ff = next(iter(ffs))
        subg = mbffg.G_clk.neighbors(ff) + [ff]
        size = len(subg)
        lib_keys.sort(key=lambda x: abs(x - size))
        # if size > 1:
        #     print(size)
        #     print(lib_keys)
        #     print(ff)
        #     exit()
        nearest = 0
        while lib_keys[nearest] > size:
            nearest += 1
        size = lib_keys[nearest]
        g = subg[:size]
        mbffg.merge_ff(",".join(g), library_seg_best[size].name)
        # ff has no neighbors
        if size > 1:
            mbffg.G_clk.remove_nodes(g)
        ffs -= set(g)


# clustering()
# mbffg.merge_ff("C1,C2,C3,C4", "FF4")
# mbffg.merge_ff("C1,C2", "FF2")
mbffg.optimize()
final_score = mbffg.scoring()
if mbffg.G.size < 1000:
    mbffg.transfer_graph_to_setting(options=options)
print(f"score: {ori_score} -> {final_score}")
# print(final_score)
mbffg.cvdraw()
