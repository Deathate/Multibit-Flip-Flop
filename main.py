import copy
import itertools
from collections import defaultdict
from pprint import pprint

import gurobipy as gp
import networkx as nx
import numpy as np
import shapely
import shapely.ops as ops
from gurobipy import GRB
from llist import dllist, sllist
from scipy.spatial.distance import cityblock
from shapely.geometry import MultiPolygon, Point, Polygon

from input import Inst, Net, PhysicalPin, read_file, visualize
from mbffg import MBFFG

input_path = "cases/sample.txt"
input_path = "cases/sampleCase"
# input_path = "v1.txt"
input_path = "v2.txt"
input_path = "cases/new_c1.txt"


# print(setting.alpha, setting.beta, setting.gamma, setting.lambde)
# pprint(setting.instances)
# pprint(setting.flip_flops)


mbffg = MBFFG(input_path)



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
    for ffidx, ff in enumerate(mbffg.get_ffs()):
        dd, ii = mbffg.min_distance_to_neightbor_inst_pin(ff.name)
        ff.metadata.moveable_area = slack_region(
            mbffg.get_pin(ii).pos,
            dd / mbffg.setting.displacement_delay,
        )
        intersection_coord = shapely.get_coordinates(ff.metadata.moveable_area)[:-1]
        rotate_45(intersection_coord)
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
        print("!!!!!!!!!!!!!!!!")
        related_ff = set()
        related_ff_ls = dllist()
        related_ff_ls_inv = defaultdict(list)
        current_node = interval_graph_x.first
        any_decision = False
        while current_node:
            ff_name, ff_is_start = current_node.value[1], current_node.value[2]
            related_ff_ls.append(current_node)
            related_ff_ls_inv[current_node.value[1]].append(related_ff_ls.last)

            print("--", ff_name, related_ff_ls.size)
            print(current_node, current_node.next)
            # found decision point
            if (
                related_ff_ls.size > 1
                and not related_ff_ls.last.value.value[2]
                and related_ff_ls.last.prev.value.value[2]
                and related_ff_ls.last.value.value[1] != related_ff_ls.last.prev.value.value[1]
            ):
                y_interval_start = False
                related_ff_y = set()
                related_ff_y_ls = dllist()
                max_clique = []
                current_node_y = interval_graph_y.first
                print(f"decision point x {ff_name}, len {len(interval_graph_x)}")
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
                            print(f"decision point y {ff_name_y}")
                            if len(new_clique := related_ff_y.intersection(related_ff)) > len(
                                max_clique
                            ):
                                max_clique = new_clique
                                print("update", max_clique)
                                if len(max_clique) >= mbffg.maximum_bits_of_library():
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
                            print("+", ff_name_y)
                        else:
                            related_ff_y.remove(ff_name_y)
                            print("-", ff_name_y)
                    current_node_y = current_node_y.next
                if len(max_clique) > 0:
                    # find appropriate library
                    B = 0
                    clique_size = sum(
                        [
                            flip_flop_list[flip_flop_names[c]].bit_number(library_list)
                            for c in max_clique
                        ]
                    )
                    max_clique.remove(ff_name)
                    for lib in library_sorted:
                        if lib.bit_number <= clique_size:
                            B = lib.bit_number
                            print(f"choose lib {lib.bit_number}")
                            break
                    Btmp = B
                    decision_point_ff = flip_flop_list[flip_flop_names[ff_name]]
                    B -= decision_point_ff.bit_number(library_list)
                    print("remain size", B)
                    k = [ff_name]

                    max_clique = list(max_clique)
                    max_clique.sort(
                        key=lambda x: flip_flop_list[flip_flop_names[x]].bit_number(library_list),
                        reverse=True,
                    )
                    for c in max_clique:
                        bit = flip_flop_list[flip_flop_names[c]].bit_number(library_list)
                        t = B - bit
                        if t >= 0:
                            print("select", c, bit)
                            B = t
                            k.append(c)
                        if t == 0:
                            break

                    if B != 0:
                        print("error")
                        for c in max_clique:
                            bit = flip_flop_list[flip_flop_names[c]].library
                            print(c, bit)
                        current_node = current_node.next
                    else:
                        any_decision = True
                        print("max_clique", k)
                        K.append({"bit": Btmp, "ff": k})
                        print("remove", k)
                        tmp = current_node.next
                        while tmp.value[1] in k:
                            tmp = tmp.next
                        current_node = tmp
                        print("current_node move to", current_node)
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
                    print("no clique found")
                    current_node = current_node.next

            else:
                current_node = current_node.next
            print("related ff")
            print(related_ff)
            if ff_is_start:
                related_ff.add(ff_name)
                print("+", ff_name)
            else:
                related_ff.remove(ff_name)
                print("-", ff_name)
        return any_decision

    run(False)


ori_score = mbffg.scoring()
mbffg.merge_ff("flip_flop8,flip_flop9", "ff2")
clustering()
exit()
mbffg.legalization()
final_score = mbffg.scoring()
print(ori_score - final_score)
mbffg.transfer_graph_to_setting(extension="html")
