import copy
import itertools
from collections import defaultdict
from pprint import pprint

import gurobipy as gp
import networkx as nx
import numpy as np
import shapely.ops as ops
from gurobipy import GRB
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
ori_score = mbffg.scoring()
mbffg.merge_ff("new_flip_flop_1,new_flip_flop_2", "ff4")
# print(mbffg.)
# exit()
# a = mbffg.get_ffs("c1")[0]
# mbffg.merge_ff("c1,c2,c3,c4", "ff4").moveto((0, 10))
mbffg.legalization()
# mbffg.merge_ff("c1,c2,c3,c4", "ff4").moveto((0, 10))
# node = "c4/d"
# print(node)
# print(mbffg.get_prev_ffs(node))
# print(mbffg.get_prev_pin(node))
# node = "c2/d"
# print(node)
# print(mbffg.get_prev_ffs(node))
# print(mbffg.get_prev_pin(node))
# print(mbffg.get_ffs()[0].dpins[0].rel_pos)


# mbffg.transfer_graph_to_setting(extension="svg")
final_score = mbffg.scoring()
print(ori_score, final_score)
