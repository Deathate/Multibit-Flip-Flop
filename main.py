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
        print(ii)
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


ori_score = mbffg.scoring()
mbffg.merge_ff("flip_flop8,flip_flop9", "ff2")
clustering()
exit()
mbffg.legalization()
final_score = mbffg.scoring()
print(ori_score - final_score)
mbffg.transfer_graph_to_setting(extension="html")
