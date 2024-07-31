import itertools
import time
from collections import defaultdict
from functools import cache, cached_property, partial
from pprint import pprint

import cv2
import gurobipy as gp

# import networkx as nx
import numpy as np
import rtree
import rustlib
from gurobipy import GRB
from scipy.spatial import KDTree

# from scipy.spatial.distance import cityblock
from tqdm.auto import tqdm

import graphx as nx
from input import Flip_Flop, Inst, Net, PhysicalPin, VisualizeOptions, read_file, visualize
from utility import *

print_tmp = print


def print(*args):
    print_tmp(*args) if len(args) > 1 else pprint(args[0]) if args else print_tmp()


def cityblock(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))


D_TAG = 2
Q_TAG = 1


class MBFFG:
    def __init__(self, file_path):
        print("Reading file...")
        setting = read_file(file_path)
        print("File read")
        G = nx.DiGraph()
        for inst in setting.instances:
            if inst.is_gt:
                in_pins = [pin.full_name for pin in inst.pins if pin.is_in]
                out_pins = [pin.full_name for pin in inst.pins if pin.is_out]
                G.add_edges_from(itertools.product(out_pins, in_pins))
            for pin in inst.pins:
                G.add_node(pin.full_name, pin=pin)
                if pin.is_q:
                    G.add_tag(pin.full_name, Q_TAG)
                elif pin.is_d:
                    G.add_tag(pin.full_name, D_TAG)
        for input in setting.inputs:
            for pin in input.pins:
                G.add_node(pin.full_name, pin=pin)
        for output in setting.outputs:
            for pin in output.pins:
                G.add_node(pin.full_name, pin=pin)
                pin.slack = 0
        for net in setting.nets:
            output_pin = net.pins[0]
            for pin in net.pins[1:]:
                G.add_edge(pin.full_name, output_pin.full_name)
        G_clk = nx.DiGraph()
        for net in setting.nets:
            if any([pin.is_clk for pin in net.pins]):
                G_clk.add_edges_from(
                    [
                        (a.inst.name, b.inst.name)
                        for a, b in itertools.permutations(net.pins, 2)
                        if a.is_clk and b.is_clk
                    ]
                )
        self.G_clk = G_clk

        print("Graph created")
        self.G = G
        self.setting = setting
        self.graph_num = 1
        # self.calculate_undefined_slack()
        pin_mapper = {}
        for node, data in G.nodes(data="pin"):
            pin_mapper[node] = data
        self.pin_mapper = pin_mapper
        print("Pin mapper created")
        # self.G = copy.deepcopy(G)
        self.G = G
        ff_filter = set()
        ffs = {}
        for node, data in self.G.nodes(data="pin"):
            if data.is_ff and data.inst.name not in ff_filter:
                ff_filter.add(data.inst.name)
                ffs[data.inst.name] = data.inst
        self.ffs = ffs
        for inst in self.setting.instances:
            if inst.is_ff:
                assert [self.get_pin(dpin).slack for dpin in inst.dpins].count(
                    None
                ) == 0, f"FF {inst.name} has None slack"
        self.pin_mapping_info = []
        print("MBFFG created")

    # def calculate_undefined_slack(self):
    #     for input in self.setting.inputs:
    #         # print(input.name)
    #         for pin in self.G[input.name]:
    #             pin_node = self.get_pin(pin)
    #             if pin_node.name.startswith("d") and pin_node.slack is None:
    #                 pin_node.slack = 0
    #                 # print(f"Set slack for {pin_node.full_name} to 0")

    #     for inst in self.setting.instances:
    #         if inst.is_ff:
    #             for pin in inst.pins:
    #                 if pin.name.startswith("d") and pin.slack is None:
    #                     paths = self.get_prev_ffs_path(pin.full_name)
    #                     path_slacks = np.zeros(len(paths))
    #                     for i, path in enumerate(paths):
    #                         ff0, ffn = path[0], path[-1]
    #                         if len(path) == 2:
    #                             path_slacks[i] = self.get_inst(ff0).qpin_delay + (
    #                                 self.setting.displacement_delay
    #                                 * (cityblock(self.get_pin(ff0).pos, self.get_pin(ffn).pos))
    #                             )
    #                         else:
    #                             c0 = path[1]
    #                             cn = path[-2]
    #                             path_slacks[i] = self.get_inst(ff0).qpin_delay + (
    #                                 self.setting.displacement_delay
    #                                 * (
    #                                     cityblock(self.get_pin(ff0).pos, self.get_pin(c0).pos)
    #                                     + cityblock(self.get_pin(c0).pos, self.get_pin(cn).pos)
    #                                     + cityblock(self.get_pin(cn).pos, self.get_pin(ffn).pos)
    #                                 )
    #                             )
    #                     pin.slack = np.max(path_slacks)

    def get_origin_inst(self, pin_name) -> Inst:
        return self.pin_mapper[pin_name].inst

    def get_origin_pin(self, pin_name):
        return self.pin_mapper[pin_name]

    def get_inst(self, pin_name):
        return self.G.nodes[pin_name]["pin"].inst

    def get_insts(self, pin_names):
        return [self.G.nodes[pin_name]["pin"].inst for pin_name in pin_names]

    def get_pin(self, pin_name) -> PhysicalPin:
        return self.G.nodes[pin_name]["pin"]

    def get_prev_ffs_path(self, node_name, temp_path=None, full_path=None):
        assert (temp_path is not None and full_path is not None) or (
            temp_path is None and full_path is None
        )
        if temp_path is None:
            temp_path = []
            full_path = []

        temp_path = temp_path + [node_name]
        node_pin = self.get_pin(node_name)
        if node_pin.is_gt:
            if node_pin.is_out:
                neightbors = [n for n in self.get_inst(node_name).inpins]
            elif node_pin.is_in:
                neightbors = []
                for n in self.G.neighbors(node_name):
                    if self.get_inst(n).name != node_pin.inst.name:
                        neightbors.append(n)
                neightbors = [
                    n
                    for n in self.G.neighbors(node_name)
                    if self.get_inst(n).name != node_pin.inst.name
                ]
            else:
                raise ValueError(f"Unknown gate type {node_pin}")
        else:
            neightbors = list(self.G.neighbors(node_name))

        for neighbor in neightbors:
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io:
                # full_path.append((temp_path + [neighbor])[::-1])
                pass
            elif neighbor_pin.is_ff:
                if neighbor_pin.is_q:
                    full_path.append((temp_path + [neighbor])[::-1])
            else:
                self.get_prev_ffs_path(neighbor, temp_path, full_path)
        return full_path

    @cache
    def c_get_prev_ffs_path(self, node_name, parent_neighbors=None):
        neightbors_ori = set(self.G.neighbors(node_name))
        neightbors = neightbors_ori - (parent_neighbors if parent_neighbors else set())
        neightbors = frozenset(neightbors)
        result = []
        for neighbor in neightbors:
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io:
                pass
            elif neighbor_pin.is_ff:
                if neighbor_pin.is_q:
                    result.append((neighbor, node_name))
            else:
                result.extend(self.c_get_prev_ffs_path(neighbor, neightbors))
        return list(set(result))

    @cache
    def prev_ffs_cache(self):
        return self.G.get_ancestor_until_map(Q_TAG, D_TAG)

    def get_prev_ffs(self, node_name):
        return self.prev_ffs_cache()[node_name]

    @cache
    def prev_pin_cache(self):
        return self.G.get_all_neighbors(D_TAG)

    def get_prev_pin(self, node_name):
        prev_pins = []
        for neighbor in self.prev_pin_cache()[node_name]:
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io or neighbor_pin.is_gt:
                prev_pins.append(neighbor)
        assert len(prev_pins) <= 1, f"Multiple previous pins for {node_name}, {prev_pins}"
        if not prev_pins:
            return None
        else:
            return prev_pins[0]

    def get_prev_inst_pin(self, node_name):
        assert self.get_pin(node_name).is_ff
        prev_pins = []
        for neighbor in self.G.neighbors(node_name):
            # neighbor_pin = self.get_pin(neighbor)
            prev_pins.append(neighbor)
        assert len(prev_pins) <= 1, f"Multiple previous pins for {node_name}, {prev_pins}"
        if not prev_pins:
            return []
        else:
            return [prev_pins[0]]

    def get_fol_inst_pins(self, node_name):
        assert self.get_pin(node_name).is_ff
        res = []
        for neighbor in self.G.neighbors(node_name):
            res.append(neighbor)
        return res

    def qpin_delay_loss(self, node_name):
        return self.get_origin_inst(node_name).qpin_delay - self.get_inst(node_name).qpin_delay

    def timing_slack(self, node_name):
        node_pin = self.get_pin(node_name)
        if node_pin.is_in or node_pin.is_gt:
            return 0
        if node_pin.is_q:
            return 0
        assert self.get_origin_pin(node_name).slack is not None, f"No slack for {node_name}"
        self_displacement_delay = 0
        prev_pin = self.get_prev_pin(node_name)
        if prev_pin:
            self_displacement_delay = (
                self.original_pin_distance(prev_pin, node_name)
                - self.current_pin_distance(prev_pin, node_name)
            ) * self.setting.displacement_delay

        prev_ffs = self.get_prev_ffs(node_name)
        prev_ffs_qpin_displacement_delay = np.zeros(len(prev_ffs) + 1)
        for i, (pff, qpin) in enumerate(prev_ffs):
            prev_ffs_qpin_displacement_delay[i] = (
                self.qpin_delay_loss(pff)
                + (self.original_pin_distance(pff, qpin) - self.current_pin_distance(pff, qpin))
                * self.setting.displacement_delay
            )
        total_delay = (
            +self.get_origin_pin(node_name).slack
            + self_displacement_delay
            + min(prev_ffs_qpin_displacement_delay)
        )

        return total_delay

    def merge_ff(self, insts: str | list, lib: str):
        if isinstance(insts, str):
            insts = self.get_ffs(insts)
        G = self.G
        pin_mapper = self.pin_mapper
        assert (
            sum([inst.lib.bits for inst in insts]) == self.setting.library[lib].bits
        ), f"FFs not match target {self.setting.library[lib].bits} bits lib, try to merge {sum([inst.lib.bits for inst in insts])} bits"
        new_inst = self.setting.get_new_instance(lib)
        new_name = "_".join([inst.name for inst in insts])
        new_inst.name += "_" + new_name

        for new_pin in new_inst.pins:
            G.add_node(new_pin.full_name, pin=new_pin)
            if new_pin.is_q:
                G.add_tag(new_pin.full_name, Q_TAG)
            elif new_pin.is_d:
                G.add_tag(new_pin.full_name, D_TAG)

        dindex, qindex = 0, 0
        for inst in insts:
            for pin in inst.pins:
                if pin.is_d:
                    dpin_fullname = new_pin.inst.dpins[dindex]
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(dpin_fullname, neightbor)
                    dindex += 1
                    pin_mapper[dpin_fullname] = pin
                    self.pin_mapping_info.append((pin.full_name, dpin_fullname))
                elif pin.is_q:
                    # qpin_fullname = f"{new_pin.inst_name}/{qpin_name}"
                    qpin_fullname = new_pin.inst.qpins[qindex]
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(qpin_fullname, neightbor)
                    qindex += 1
                    pin_mapper[qpin_fullname] = pin
                    self.pin_mapping_info.append((pin.full_name, qpin_fullname))
                else:
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(new_pin.full_name, neightbor)
                    pin_mapper[new_pin.full_name] = pin
                    self.pin_mapping_info.append((pin.full_name, new_pin.full_name))
                G.remove_node(pin.full_name)
            del self.ffs[inst.name]

        self.ffs[new_inst.name] = new_inst
        return new_inst

    def get_ffs(self, ff_names=None) -> list[Inst]:
        if ff_names is None:
            return list(self.ffs.values())
        else:
            if isinstance(ff_names, str):
                ff_names = ff_names.split(",")
            try:
                return [self.ffs[ff_name] for ff_name in ff_names]
            except:
                assert False, f"FFs {ff_names} not found"

    def get_ff(self, ff_name) -> Inst:
        return self.ffs[ff_name]

    def get_ffs_names(self):
        return tuple(ff.name for ff in self.get_ffs())

    def get_gates(self):
        return [inst for inst in self.setting.instances if not inst.is_ff]

    def get_library(self):
        return self.setting.library

    @cache
    def maximum_bits_of_library(self):
        return max([lib.bits for lib in self.setting.library.values()])

    def scoring(self):
        print("Scoring...")
        total_tns = 0
        total_power = 0
        total_area = 0
        for ff in tqdm(self.get_ffs()):
            slacks = [min(self.timing_slack(dpin), 0) for dpin in ff.dpins]
            total_tns += sum(slacks)
            total_power += ff.lib.power
            total_area += ff.lib.area
        print("Scoring done")
        return (
            -self.setting.alpha * total_tns
            + self.setting.beta * total_power
            + self.setting.gamma * total_area
        )

    def original_pin_distance(self, node1, node2):
        return cityblock(self.get_origin_pin(node1).pos, self.get_origin_pin(node2).pos)

    def current_pin_distance(self, node1, node2):
        return cityblock(self.get_pin(node1).pos, self.get_pin(node2).pos)

    def transfer_graph_to_setting(self, options, visualized=True, show_distance=False):
        if len(self.setting.instances) > 1000:
            extension = "pdf"
        else:
            extension = "html"
        G = self.G
        setting = self.setting
        setting.instances = []
        instance_names = set()
        for name, data in G.nodes(data="pin"):
            if data.is_io:
                continue
            elif data.inst.name not in instance_names:
                setting.instances.append(data.inst)
                instance_names.add(data.inst.name)

        setting.nets = []
        for node1, node2 in G.edges():
            if self.get_inst(node1) == self.get_inst(node2) and not self.get_pin(node1).is_ff:
                continue
            net = Net("n", num_pins=2)
            net.pins = [self.get_pin(node1), self.get_pin(node2)]
            if show_distance:
                net.metadata = cityblock(net.pins[0].pos, net.pins[1].pos)
            setting.nets.append(net)
        if visualized:
            visualize(
                setting,
                options,
                file_name=f"output{self.graph_num}.{extension}",
                resolution=None if extension == "html" else 10000,
            )
            self.graph_num += 1

    def print_graph(G):
        for node, data in G.nodes(data="pin"):
            print(node, list(G.neighbors(node)))

    def reset_cache(self):
        self.prev_ffs_cache.cache_clear()
        self.prev_pin_cache.cache_clear()

    def optimize(self):
        self.reset_cache()

        def cityblock_variable(model, v1, v2, bias, weight, intercept):
            delta_x = model.addVar(lb=-GRB.INFINITY)
            delta_y = model.addVar(lb=-GRB.INFINITY)
            abs_delta_x = model.addVar()
            abs_delta_y = model.addVar()
            cityblock_distance = model.addVar(lb=-GRB.INFINITY)
            model.addConstr(delta_x == v1[0] - v2[0])
            model.addConstr(delta_y == v1[1] - v2[1])
            model.addConstr(abs_delta_x == gp.abs_(delta_x))
            model.addConstr(abs_delta_y == gp.abs_(delta_y))
            model.addLConstr(
                cityblock_distance == weight * (abs_delta_x + abs_delta_y + bias) + intercept
            )
            return cityblock_distance

        print("Optimizing...")
        with gp.Env(empty=True) as env:
            env.setParam("LogToConsole", 1)
            env.start()
            with gp.Model(env=env) as model:
                # model.setParam(GRB.Param.Presolve, 1)
                # model.Params.Presolve = 2
                ff_vars = {}
                for ff in self.get_ffs():
                    ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                        name=ff.name + "1"
                    )
                # dis2ori_locations = []
                negative_slack_vars = []
                for ff in tqdm(self.get_ffs()):
                    for curpin in ff.dpins:
                        ori_slack = self.get_origin_pin(curpin).slack
                        prev_pin = self.get_prev_pin(curpin)
                        if prev_pin:
                            current_pin = self.get_pin(curpin)
                            current_pin_pos = [
                                a + b
                                for a, b in zip(ff_vars[current_pin.inst.name], current_pin.rel_pos)
                            ]
                            dpin_pin = self.get_pin(prev_pin)
                            dpin_pin_pos = dpin_pin.pos
                            ori_distance = self.original_pin_distance(prev_pin, curpin)
                            prev_pin_displacement_delay = cityblock_variable(
                                model,
                                current_pin_pos,
                                dpin_pin_pos,
                                -ori_distance,
                                self.setting.displacement_delay,
                                0,
                            )
                        else:
                            prev_pin_displacement_delay = 0
                        displacement_distances = []
                        prev_ffs = self.get_prev_ffs(curpin)
                        for pff, qpin in prev_ffs:
                            pff_pin = self.get_pin(pff)
                            qpin_pin = self.get_pin(qpin)
                            pff_pos = [
                                a + b for a, b in zip(ff_vars[pff_pin.inst.name], pff_pin.rel_pos)
                            ]
                            if qpin_pin.is_ff:
                                qpin_pos = [
                                    a + b
                                    for a, b in zip(ff_vars[qpin_pin.inst.name], qpin_pin.rel_pos)
                                ]
                            else:
                                qpin_pos = qpin_pin.pos
                            ori_distance = self.original_pin_distance(pff, qpin)
                            distance_var = cityblock_variable(
                                model,
                                pff_pos,
                                qpin_pos,
                                -ori_distance,
                                self.setting.displacement_delay,
                                -self.qpin_delay_loss(pff),
                            )
                            displacement_distances.append(distance_var)
                        if len(displacement_distances) > 0:
                            min_displacement_distance = model.addVar(lb=-GRB.INFINITY)
                            model.addConstr(
                                min_displacement_distance == gp.min_(displacement_distances)
                            )
                        else:
                            min_displacement_distance = 0

                        slack_var = model.addVar(name=curpin, lb=-GRB.INFINITY)
                        model.addConstr(
                            slack_var
                            == ori_slack - (prev_pin_displacement_delay + min_displacement_distance)
                        )
                        negative_slack_var = model.addVar(
                            name=f"negative_slack for {curpin}", lb=-GRB.INFINITY
                        )
                        model.addConstr(negative_slack_var == gp.min_(slack_var, 0))
                        negative_slack_vars.append(negative_slack_var)

                    # dis2ori = cityblock_variable(model, ff_vars[ff.name], ff.pos, 0, 1, 0)
                    # dis2ori_locations.append(dis2ori)
                model.setObjective(-gp.quicksum(negative_slack_vars))
                # model.setObjectiveN(-gp.quicksum(min_negative_slack_vars), 0, priority=1)
                # model.setObjectiveN(gp.quicksum(dis2ori_locations), 1, priority=0)
                model.optimize()

                for name, ff_var in ff_vars.items():
                    self.get_ffs(name)[0].moveto((ff_var[0].X, ff_var[1].X))
                    ff_vars[name] = self.get_ff(name).pos
        self.legalization_rust(ff_vars)
        # self.legalization(ff_vars)

    def legalization(self, ff_vars):
        points = []
        for placement_row in self.setting.placement_rows:
            for i in range(placement_row.num_cols):
                x, y = placement_row.x + i * placement_row.width, placement_row.y
                points.append((x, y))

        def generator_function(somedata):
            for i, obj in enumerate(somedata):
                rect = (obj[0], obj[1], obj[2], obj[3])
                yield (
                    i,
                    rect,
                    rect,
                )

        p = rtree.index.Property(leaf_capacity=1000, fill_factor=0.9)
        idx = rtree.index.Index(
            (
                generator_function([gate.box.buffer(-0.01).bounds for gate in self.get_gates()])
                if self.get_gates()
                else []
            ),
            property=p,
        )
        # gate_box = [gate.box for gate in self.get_gates()]
        # strtree = shapely.STRtree(gate_box)

        points = np.ma.array(points, mask=False)

        # tree = KDTree(points)

        # def remove_points_from_ma(points, tree, gates):
        #     points_within_gate = tree.query_ball_point(
        #         [gate.center for gate in gates], r=[gate.diag_l2 / 2 for gate in gates], p=2
        #     )
        #     for i in range(len(gates)):
        #         index_with_box = [
        #             j
        #             for j in points_within_gate[i]
        #             if Point(points[j]).within(gates[i].box)
        #             or (cityblock(points[j], gates[i].ll) < 1e-2)
        #         ]
        #         points[index_with_box] = np.ma.masked

        # remove_points_from_ma(points, tree, self.get_gates())
        print("Replace illegal position...")
        remaining_ffs = set(ff_vars.keys())
        with tqdm(total=len(remaining_ffs)) as pbar:
            while remaining_ffs:
                points = points[~points.mask[:, 0]]
                tree = KDTree(points)
                placed_ffs = []
                for name in remaining_ffs.copy():
                    ff_var = ff_vars[name]
                    dd, ii = tree.query([ff_var], k=100)
                    for i in ii[0]:
                        if np.ma.is_masked(points[i]):
                            continue
                        ff = self.get_ffs(name)[0]
                        ff.moveto(points[i])
                        try:
                            overlapped_ff = idx.count(ff.bbox)
                        except:
                            print(ff)
                            print(ff.bbox)
                            print(ff.box)
                        if overlapped_ff == 0:
                            idx.insert(i, ff.box.buffer(-0.01).bounds)
                            remaining_ffs.remove(name)
                            placed_ffs.append(name)
                            pbar.update(1)
                            break
                        else:
                            points[i] = np.ma.masked

    def legalization_rust(self, ff_vars):
        print("Legalizing...")
        aabbs = []
        points = []
        for placement_row in self.setting.placement_rows:
            for i in range(placement_row.num_cols):
                x, y = placement_row.x + i * placement_row.width, placement_row.y
                points.append((x, y))
                aabbs.append(((x, y), (x + placement_row.width, y + placement_row.height)))
        barriers = [(gate.ll, gate.ur) for gate in self.get_gates()]
        ff_names = list(ff_vars.keys())
        candidates = list([self.get_ff(x).bbox_corner for x in ff_names])
        candidates.sort(key=lambda x: (x[1][0] - x[0][0]) * (x[1][1] - x[0][1]))
        result, size = rustlib.legalize(aabbs, barriers, candidates)
        assert size == len(candidates)
        for i, name in enumerate(ff_names):
            ff = self.get_ff(name)
            ff.moveto(result[i])

    def output(self, path):
        with open(path, "w") as file:
            file.write(f"CellInst {len(self.get_ffs())}\n")
            for ff in self.get_ffs():
                file.write(f"Inst {ff.name} {ff.lib.name} {ff.pos[0]} {ff.pos[1]}\n")
            for f, t in self.pin_mapping_info:
                file.write(f"{f} map {t}\n")

    @static_vars(graph_num=1)
    def cvdraw(self):
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        BLACK = (0, 0, 0)
        img_width = self.setting.die_size.xUpperRight
        img_height = self.setting.die_size.yUpperRight
        ratio = 6000 / max(img_width, img_height)
        img_width, img_height = int(img_width * ratio), int(img_height * ratio)
        img = np.ones((img_height, img_width, 3), np.uint8) * 255
        for placement_row in self.setting.placement_rows:
            x, y = placement_row.x, placement_row.y
            w = placement_row.width * placement_row.num_cols
            h = placement_row.height
            x, y, w, h = int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)
            img = cv2.line(img, (x, y), (x + w, y), RED, 1)
            for i in range(1, placement_row.num_cols + 1):
                x = placement_row.x + i * placement_row.width
                x = int(x * ratio)
                img = cv2.line(img, (x, y), (x, y + h), RED, 1)
            # w, h = (
            #     placement_row.width * placement_row.num_cols,
            #     placement_row.height,
            # )
            # x, y, w, h = int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio)
            # img = cv2.line(img, (x, y), (x + w, y), RED, 1)
            # w, h = (
            #     placement_row.width * placement_row.num_cols,
            #     placement_row.height,
            # )
        for ff in self.get_ffs():
            x, y = ff.pos
            w = ff.width
            h = ff.height
            x, y = int(x * ratio), int(y * ratio)
            w, h = int(w * ratio), int(h * ratio)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), BLACK, -1)
        for gate in self.get_gates():
            x, y = gate.ll
            w, h = gate.width, gate.height
            x, y = int(x * ratio), int(y * ratio)
            w, h = int(w * ratio), int(h * ratio)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), BLUE, -1)
        img = cv2.flip(img, 0)
        file_name = f"output{MBFFG.cvdraw.graph_num}.png"
        MBFFG.cvdraw.graph_num += 1
        cv2.imwrite(file_name, img)
        print(f"Image saved to {file_name}")

    def utility_ratio(self):
        a = self.setting.die_size.area
        b = sum(map(lambda x: x.area, self.get_gates()))
        c = sum(map(lambda x: x.area, self.get_ffs()))
        return (b + c) / a

    @cache
    def get_selected_library(self):
        library_sorted = sorted(
            self.get_library().values(),
            key=lambda x: (x.power * self.setting.beta + x.area * self.setting.gamma) / x.bits,
        )
        library_seg_best: dict[int, Flip_Flop] = {}
        for lib in library_sorted:
            if lib.bits not in library_seg_best:
                library_seg_best[lib.bits] = lib
        lib_keys = list(library_seg_best.keys())
        return library_seg_best, lib_keys


def get_pin_name(node_name):
    return node_name.split("/")[1]


def get_pin_name(node_name):
    return node_name.split("/")[1]
