import math
import signal
import sys
import traceback
from bisect import bisect_left
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from pprint import pprint

import numpy as np
import rustlib
import shapely
from llist import dllist, sllist
from rtree.index import Index
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from sklearn.neighbors import NearestNeighbors

import graphx as nx
from faketime_utl import ensure_time
from mbffg import D_TAG, MBFFG, Q_TAG, VisualizeOptions
from utility import *

signal.signal(signal.SIGINT, signal.SIG_DFL)
from input import *


# @blockPrinting
def main(step_options):
    # ensure_time()
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
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

        input_path = "cases/testcase0.txt"
        input_path = "cases/sample_exp.txt"
        input_path = "cases/sample_exp_mbit.txt"
        input_path = "cases/sample_exp_comb.txt"
        input_path = "cases/sample_exp_comb2.txt"
        input_path = "cases/sample_exp_comb3.txt"
        input_path = "cases/sample_exp_comb4.txt"
        input_path = "cases/sample_exp_comb5.txt"
        input_path = "cases/sample.txt"
        input_path = "cases/testcase2_0812.txt"
        input_path = "cases/testcase1_0812.txt"

        os.system(f"./symlink.sh {input_path}")
    options = VisualizeOptions(
        line=True,
        cell_text=True,
        io_text=True,
        placement_row=True,
    )
    mbffg = MBFFG(input_path)
    # score:1028974779.12962
    # mbffg.output(output_path)
    # exit()
    mbffg.transfer_graph_to_setting(options=options)
    mbffg.cvdraw("output/1_initial.png")
    # mbffg.get_pin("C3/D").inst.r_moveto((-6, 0))
    # mbffg.get_pin("C6/D").inst.r_moveto((8, -10))
    # mbffg.get_ff("C2").r_moveto((-4, 0))
    # mbffg.merge_ff("C7", "FF1_1", 0)
    # mbffg.merge_ff("C2", "FF2_1", 0)
    # mbffg.merge_ff("C1,C3", "FF2", 0)
    # mbffg.demerge_ff("C2", "FF1")

    use_linear_sum_assignment = False

    def centroid(inst_name) -> tuple[float, float]:
        return np.mean(
            [mbffg.get_origin_pin(x).inst.pos for x in mbffg.get_ff(inst_name).dpins], axis=0
        )

    def potential_space_cluster_detail(
        arranged_library_name, potential_space_dict, ffs, rtree, debug, allow_dis
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
        for ff in tqdm(ffs_order):
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

            average_pos = centroid(ff)
            # if debug:
            #     print(average_pos)
            #     print(current_inst)
            #     print(current_inst.dpins)
            #     exit()
            if size > 1:
                result = []
                result_bits = 0
                while True:
                    if neigh_tree.size() == 0:
                        print("no enough space", current_lib.bits)
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
                new_pos = average_pos
                potential_space_rtree = potential_space_dict[selected_lib.name]
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
        # print(arranged_library_name)
        # print([mbffg.get_library(x).bits for x in arranged_library_name])
        # exit()
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

        arranged_library_name.sort(
            key=lambda x: mbffg.get_library(x).bits if mbffg.get_library(x).bits > 1 else 100
        )
        # print(arranged_library_name)
        # print(arranged_library_size.keys())
        # print(arranged_library_name_part)
        # print(one_bit_library)
        # exit()
        rtree = rustlib.Rtree()
        arranged_library_name_part = arranged_library_name[:1]
        adoptees, orphans = potential_space_cluster_detail(
            arranged_library_name_part,
            potential_space_dict,
            set([x.name for x in mbffg.get_ffs()]),
            rtree,
            False,
            allow_dis=None,
        )
        arranged_library_name_part = arranged_library_name[1:2]
        potential_space_cluster_detail(
            arranged_library_name_part,
            potential_space_dict,
            set(adoptees),
            rtree,
            True,
            allow_dis=None,
        )
        one_bit_librarys = list(
            filter(lambda x: mbffg.get_library(x).bits == 1, arranged_library_name)
        )
        adoptees, orphans = potential_space_cluster_detail(
            one_bit_librarys,
            potential_space_dict,
            set([x.name for x in mbffg.get_ffs() if x.bits == 1]),
            rtree,
            True,
            allow_dis=math.inf,
        )

        mbffg.reset_cache()

    # mbffg.cvdraw("output/3_cluster.png")
    if step_options[4]:
        print("legalization")
        ffs_order = mbffg.get_ffs_names()
        _, ff_util = mbffg.utilization_score()
        ffs_order.sort(
            key=lambda x: (
                -mbffg.get_ff(x).area,
                ff_util[x][0],
                -ff_util[x][1],
                -ff_util[x][2],
                mbffg.get_ff(x).pos,
            ),
            reverse=True,
        )
        target = [centroid(x) for x in ffs_order]
        candidates = [mbffg.get_ff(x).bbox_corner_true for x in ffs_order]
        result = rustlib.finetune(
            mbffg.row_box(),
            obstacles,
            candidates,
            target,
            mbffg.setting.die_size.bbox_corner,
        )
        for i, ff in enumerate(ffs_order):
            mbffg.get_ff(ff).moveto(result[i])
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
    mbffg.transfer_graph_to_setting(options=options)
    mbffg.output(output_path)


# scoring, demerge, optimize, cluster, legalization
main([0, 1, 0, 1, 1])
# main([1, 1, 1, 0])

# for step_options in product([True, False], repeat=4):
#     if step_options[0] == False and step_options[1] == True:
#         continue
#     print(step_options)
#     with HiddenPrints():
#         score = main([True, True, True, True])
#     print(score)
