import math
import signal
import sys
from itertools import chain

import numpy as np
import rustlib
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

import utility
from mbffg import MBFFG, VisualizeOptions
from utility import *

signal.signal(signal.SIGINT, signal.SIG_DFL)
from input import *

if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    DEBUG = True
    utility.DEBUG = False
else:
    output_path = "output/output.txt"
    input_path = "cases/new_c5.txt"
    input_path = "cases/new_c1.txt"
    input_path = "cases/new_c3.txt"
    input_path = "cases/testcase1.txt"
    input_path = "cases/v2.txt"
    input_path = "cases/testcase1_balanced.txt"
    input_path = "cases/testcase1.txt"
    input_path = "cases/testcase1_0614.txt"

    input_path = "cases/sample_exp_comb3.txt"
    input_path = "cases/sample_exp_comb4.txt"
    input_path = "cases/sample_exp_comb5.txt"
    input_path = "cases/testcase0.txt"
    input_path = "cases/sample_exp_mbit.txt"
    input_path = "cases/sample_exp_comb.txt"
    input_path = "cases/sample_exp_comb2.txt"
    input_path = "cases/testcase3.txt"
    input_path = "cases/testcase2_0812.txt"
    input_path = "cases/testcase1_0812.txt"
    input_path = "cases/sample.txt"
    input_path = "cases/sample_exp.txt"

    os.system(f"./symlink.sh {input_path}")


def main(step_options, library_index):
    options = VisualizeOptions(
        line=True,
        cell_text=True,
        io_text=True,
        placement_row=True,
    )
    mbffg = MBFFG(input_path)
    # mbffg.output(output_path)
    # exit()
    if DEBUG:
        mbffg.transfer_graph_to_setting(options=options)
        mbffg.cvdraw("output/1_initial.png")
    exit()
    use_linear_sum_assignment = False

    def potential_space_cluster_detail(
        arranged_library_name, potential_space_dict, ffs, rtree, debug, allow_dis, stable
    ):
        dist_tree = NestedDict()
        for current_inst in mbffg.get_ffs():
            if current_inst.name not in ffs:
                continue
            net = current_inst.clk_neighbor
            net_name = mbffg.get_pin(current_inst.clkpin).net_name
            if net_name in dist_tree:
                continue
            net_inst = [mbffg.get_ff(x) for x in net if x in ffs]
            tree = rustlib.Rtree()
            tree.bulk_insert([(x.pos, x.pos) for x in net_inst])
            dist_tree[net_name]["tree"] = tree
            dist_tree[net_name]["name_query"] = {x.pos: x.name for x in net_inst}

        library_sizes = [mbffg.get_library(x).bits for x in arranged_library_name]
        _, ff_util = mbffg.utilization_score()
        ffs_order = list(ffs)
        ffs_order.sort(
            key=lambda x: (
                ff_util[x][0],
                -ff_util[x][1],
                -ff_util[x][2],
                mbffg.get_ff(x).pos,
            ),
            reverse=True,
        )
        adoptees = []
        orphans = []
        for ff in tqdm(ffs_order) if DEBUG else ffs_order:
            if ff not in ffs:
                # if debug:
                #     print(f"skip {ff}")
                continue
            current_inst = mbffg.get_ff(ff)
            net = current_inst.clk_neighbor
            net_name = mbffg.get_pin(current_inst.clkpin).net_name
            subg = [x for x in net if x in ffs]
            net.clear()
            net.extend(subg)
            current_lib = current_inst.lib
            lib_idx = 0
            size = library_sizes[lib_idx]
            selected_lib = mbffg.get_library(arranged_library_name[lib_idx])
            if allow_dis is None:
                allow_dis = (
                    current_lib.power * selected_lib.bits - selected_lib.power
                ) * mbffg.setting.beta + (
                    current_lib.area * selected_lib.bits - selected_lib.area
                ) * mbffg.setting.gamma
                allow_dis /= mbffg.setting.alpha
                allow_dis += current_lib.qpin_delay - selected_lib.qpin_delay
                if allow_dis < 0:
                    continue
                allow_dis /= mbffg.setting.displacement_delay
                allow_dis /= 2

            buffer = []
            neigh = dist_tree[net_name]
            neigh_tree = neigh["tree"]

            def centroid(inst_name):
                return np.mean(
                    [mbffg.get_origin_pin(x).pos for x in mbffg.get_ff(inst_name).dpins], axis=0
                )

            average_pos = centroid(ff)
            if size > 1:
                result = []
                result_bits = 0
                while True:
                    if neigh_tree.size() == 0:
                        # print("no enough space", current_lib.bits)
                        break
                    buffer.append(neigh_tree.pop_nearest(current_inst.pos))
                    inst_name = neigh["name_query"][tuple(buffer[-1][0])]
                    if inst_name not in ffs:
                        buffer.pop()
                        continue
                    dist = cityblock(average_pos, buffer[-1][0])
                    if dist > allow_dis:
                        break
                    else:
                        result.append(inst_name)
                        result_bits += mbffg.get_ff(inst_name).bits
                        if result_bits == size:
                            break

                g = result

                if use_linear_sum_assignment:
                    inst_pin_pos = [mbffg.get_ff(x).center for x in g]
                    lib_pin_pos = [dpin.pos for dpin in selected_lib.dpins]
                    row_ind, col_ind = linear_sum_assignment(
                        distance_matrix(inst_pin_pos, lib_pin_pos, p=1)
                    )
                    g = list(
                        map(
                            lambda x: x[0],
                            sorted(list(zip(g, row_ind, col_ind)), key=lambda x: (x[1], x[2])),
                        )
                    )
            else:
                g = [ff]
            find_legal = False
            if sum([mbffg.get_ff(x).bits for x in g]) == size:
                new_pos = np.mean([centroid(x) for x in g], axis=0)
                potential_space_rtree = potential_space_dict[selected_lib.name]
                if stable:
                    for x in g:
                        rtree.delete(*mbffg.get_ff(x).bbox_corner)
                while potential_space_rtree.size() > 0:
                    point = potential_space_rtree.pop_nearest(new_pos)
                    if cityblock(point[0], new_pos) > allow_dis:
                        break
                    attempt_box = BoxContainer(
                        selected_lib.width, selected_lib.height, point[0]
                    ).bbox
                    if not mbffg.setting.die_size.inside(*attempt_box):
                        continue
                    if rtree.count(*attempt_box) == 0:
                        # if debug:
                        #     print(f"merge {g} to {selected_lib.name}")
                        inst = mbffg.merge_ff(g, selected_lib.name, lib_idx)
                        inst.moveto(point[0])
                        inst.clk_neighbor = net
                        adoptees.append(inst.name)
                        rtree.insert(*inst.bbox_corner)
                        potential_space_rtree.delete(*attempt_box)
                        ffs -= set(g)
                        find_legal = True
                        break
            else:
                if debug:
                    print(f"failed to merge {g} to {selected_lib.name}")
            if not find_legal:
                for x in buffer:
                    neigh_tree.insert(*x)
                orphans.append(ff)
                if stable:
                    for x in g:
                        rtree.insert(*mbffg.get_ff(x).bbox_corner)

        for adopt in adoptees:
            mbffg.get_ff(adopt).clk_neighbor.append(adopt)
        orphans = set(orphans).intersection(ffs)
        return adoptees, orphans

    if step_options[0]:
        ori_score, ori_stat = mbffg.scoring()
        # print(f"original score: {ori_score}")
    if step_options[1]:
        mbffg.demerge_ffs()
    if step_options[2]:
        # replace_ff_with_local_optimal()
        mbffg.optimize(global_optimize=False)
    # mbffg.cvdraw("output/2_optimize.png")
    if step_options[3]:
        # clustering_random()
        print("potential_space_cluster")
        library_classified, arranged_library_name, library_order, library_costs = (
            mbffg.sort_library_by_cost()
        )

        grid_sizes = [mbffg.get_library(x).size for x in arranged_library_name]
        row_coordinates = mbffg.row_coordinates()

        obstacles = mbffg.get_gates_box()
        potential_space_detail = rustlib.calculate_potential_space_detail(
            row_coordinates, obstacles, grid_sizes
        )
        potential_space_dict = {}
        for i, x in enumerate(arranged_library_name):
            t = rustlib.Rtree()
            points_box = []
            for point in chain.from_iterable(potential_space_detail[i]):
                points_box.append([point, point])
            t.bulk_insert(points_box)
            potential_space_dict[x] = t
        smallest_library = min(library_order)
        library_order.remove(smallest_library)

        rtree = rustlib.Rtree()
        arranged_library_name_part = [
            arranged_library_name[
                index(
                    arranged_library_name,
                    lambda x: mbffg.get_library(x).bits == library_order[library_index],
                )
            ]
        ]
        potential_space_cluster_detail(
            arranged_library_name_part,
            potential_space_dict,
            set([x.name for x in mbffg.get_ffs()]),
            rtree,
            False,
            allow_dis=None,
            stable=False,
        )

        one_bit_librarys = [
            arranged_library_name[
                index(
                    arranged_library_name, lambda x: mbffg.get_library(x).bits == smallest_library
                )
            ]
        ]
        potential_space_cluster_detail(
            one_bit_librarys,
            potential_space_dict,
            set([x.name for x in mbffg.get_ffs() if x.bits == smallest_library]),
            rtree,
            True,
            allow_dis=math.inf,
            stable=False,
        )
        mbffg.reset_cache()

    # mbffg.cvdraw("output/3_cluster.png")
    if step_options[4]:
        print("legalization")
        mbffg.legalization_rust(False)
        # mbffg.legalization_check()
    if DEBUG:
        mbffg.cvdraw("output/4_legalization.png")

    # # clustering()
    # # mbffg.merge_ff("C1,C2,C3,C4", "FF4")
    # # mbffg.merge_ff("C1,C2", "FF2")
    if step_options[0]:
        final_score, final_stat = mbffg.scoring()
        print(
            f"original score: {ori_score}, final score: {final_score}, diff: {final_score - ori_score}"
        )
        mbffg.show_statistics(ori_stat, final_stat)
    if DEBUG:
        mbffg.transfer_graph_to_setting(options=options)
    return mbffg, library_order


# scoring, demerge, optimize, cluster, legalization
min_score = math.inf
mbffg, library_order = main([0, 1, 0, 1, 0], 0)
min_mbffg = None
score = mbffg.scoring()[0]
if score < min_score:
    min_score = score
    min_mbffg = mbffg
    print(score)
for i in range(len(library_order) - 1):
    print(i)
    mbffg, library_order = main([0, 1, 0, 1, 0], i + 1)
    score = mbffg.scoring()[0]
    if score < min_score:
        min_score = score
        min_mbffg = mbffg
        print(score)
min_mbffg.output(output_path)
