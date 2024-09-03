import math
import signal
import sys
import traceback
from collections import defaultdict
from operator import itemgetter
from pprint import pprint

import numpy as np
import rustlib
import shapely
from llist import dllist, sllist
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
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
        input_path = "cases/testcase1_0812.txt"
        input_path = "cases/testcase2_0812.txt"
        input_path = "cases/sample_exp_comb.txt"
        input_path = "cases/sample_exp_mbit.txt"
        input_path = "cases/sample_exp_comb2.txt"
        input_path = "cases/sample.txt"
        input_path = "cases/sample_exp.txt"
        input_path = "cases/sample_exp_comb4.txt"
        input_path = "cases/sample_exp_comb3.txt"

        os.system(f"./symlink.sh {input_path}")
    options = VisualizeOptions(
        line=True,
        cell_text=True,
        io_text=True,
        placement_row=True,
    )
    mbffg = MBFFG(input_path)

    mbffg.transfer_graph_to_setting(options=options)
    mbffg.cvdraw("output/1_initial.png")
    # mbffg.get_pin("C3/D").inst.r_moveto((-6, 0))
    # mbffg.get_pin("C6/D").inst.r_moveto((8, -10))
    # mbffg.get_ff("C2").r_moveto((-4, 0))
    # mbffg.merge_ff("C7", "FF1_1", 0)
    # mbffg.merge_ff("C2", "FF2_1", 0)
    # mbffg.merge_ff("C1,C3", "FF2", 0)
    # mbffg.demerge_ff("C2", "FF1")

    ori_score, ori_stat = mbffg.scoring()
    # print(f"original score: {ori_score}")
    # mbffg.output(output_path)
    # exit()

    use_knn = True
    use_linear_sum_assignment = False
    use_linear_sum_assignment = min(use_knn, use_linear_sum_assignment)

    def potential_space_cluster():
        # print(mbffg.ff_stats_with_name())
        # optimal_library_segments, library_sizes = mbffg.get_selected_library()
        # total_bit = sum(bit * count for bit, count in mbffg.ff_stats())
        library_classified, cost_sorted_library, library_order, library_costs = (
            mbffg.sort_library_by_cost()
        )
        # arranged_library_name = [library_classified[x].popleft().name for x in library_order]
        row_coordinates = [
            x.get_rows() for x in sorted(mbffg.setting.placement_rows, key=lambda x: x.y)
        ]
        obstacles = [x.bbox_corner for x in mbffg.get_gates()]

        # Arrange library names based on size
        arranged_library_name = []
        arranged_library_size = set()
        for csl in cost_sorted_library:
            size = csl.size
            if size not in arranged_library_size:
                if all([(size[0] < x[0] or size[1] < x[1]) for x in arranged_library_size]):
                    arranged_library_name.append(csl.name)
                    arranged_library_size.add(size)
        # print(arranged_library_name)
        # print([x.name for x in cost_sorted_library])
        # exit()
        grid_sizes = [mbffg.get_library(x).size for x in arranged_library_name]
        potential_space = rustlib.calculate_potential_space(row_coordinates, obstacles, grid_sizes)
        library_sizes = [mbffg.get_library(x).bits for x in arranged_library_name]
        # print(potential_space)
        # print(library_sizes)
        ffs = set([x.name for x in mbffg.get_ffs()])
        ffs_order = list(ffs)
        ffs_order.sort(
            key=lambda x: (
                len(mbffg.get_ff(x).clk_neighbor),
                mbffg.get_ff(x).x,
                mbffg.get_ff(x).y,
            ),
            reverse=True,
        )

        while len(ffs_order) > 0:
            # print(potential_space, len(ffs))
            ff = ffs_order.pop()
            if ff not in ffs:
                # print(f"skip {ff}")
                continue
            net = mbffg.get_ff(ff).clk_neighbor
            subg = [ff] + [x for x in net if x in ffs and x != ff]
            size = len(subg)
            lib_idx = index(
                list(enumerate(library_sizes)),
                lambda x: x[1] <= size and potential_space[x[0]] > 0,
            )

            if lib_idx is not None:
                # there are enough potential space
                potential_space[lib_idx] -= 1
            else:
                # try to find other library
                disabled_lib_idx = set()
                lib_idx = index(
                    list(enumerate(library_sizes)),
                    lambda x: x[1] <= size and potential_space[x[0]] == 0,
                )

                while True:
                    lib_idx_sub = index(
                        list(enumerate(library_sizes)),
                        lambda x: x[1] > size
                        and potential_space[x[0]] > 0
                        and x[0] not in disabled_lib_idx,
                    )
                    if lib_idx_sub is None:
                        print("potential_space is not enough")
                        break
                    occupied_width_ratio = math.ceil(
                        mbffg.get_library(arranged_library_name[lib_idx]).width
                        / mbffg.get_library(arranged_library_name[lib_idx]).width
                    )
                    occupied_height_ratio = math.ceil(
                        mbffg.get_library(arranged_library_name[lib_idx]).height
                        / mbffg.get_library(arranged_library_name[lib_idx]).height
                    )
                    occupied_cell_ratio = 1 / (occupied_width_ratio * occupied_height_ratio)
                    if potential_space[lib_idx_sub] >= occupied_cell_ratio:
                        potential_space[lib_idx_sub] -= occupied_cell_ratio
                        break
                    else:
                        disabled_lib_idx.add(lib_idx_sub)

            size = library_sizes[lib_idx]
            selected_lib = mbffg.get_library(arranged_library_name[lib_idx])
            if use_knn:
                if size > 1:
                    neigh = NearestNeighbors()
                    neigh.fit([mbffg.get_ff(x).center for x in subg])
                    result = neigh.kneighbors(
                        [mbffg.get_ff(ff).center], n_neighbors=size, return_distance=False
                    )
                    g = itemgetter(*result[0])(subg)
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
                    g = subg[:1]
            else:
                g = subg[:size]
            # print(g, optimal_library_segments[size].name)
            mbffg.merge_ff(g, selected_lib.name, lib_idx)
            # print(f"merge {g} to {optimal_library_segments[size].name}")
            ffs -= set(g)
        mbffg.reset_cache()

    if step_options[0]:
        mbffg.demerge_ffs()
    if step_options[1]:
        # replace_ff_with_local_optimal()
        mbffg.optimize(global_optimize=False)
    # mbffg.cvdraw("output/2_optimize.png")
    if step_options[2]:
        # clustering_random()
        print("potential_space_cluster")
        potential_space_cluster()
    # mbffg.cvdraw("output/3_cluster.png")
    if step_options[3]:
        print("legalization")
        mbffg.legalization_rust(False)
        # mbffg.legalization_check()
    mbffg.cvdraw("output/4_legalization.png")

    # # clustering()
    # # mbffg.merge_ff("C1,C2,C3,C4", "FF4")
    # # mbffg.merge_ff("C1,C2", "FF2")

    final_score, final_stat = mbffg.scoring()
    mbffg.show_statistics(ori_stat, final_stat)
    mbffg.transfer_graph_to_setting(options=options)
    mbffg.output(output_path)
    return final_score


# demerge, optimize, cluster, legalization
# main([0, 0, 0, 0])
main([1, 0, 1, 1])
# main([1, 1, 1, 0])

# for step_options in product([True, False], repeat=4):
#     if step_options[0] == False and step_options[1] == True:
#         continue
#     print(step_options)
#     with HiddenPrints():
#         score = main([True, True, True, True])
#     print(score)
