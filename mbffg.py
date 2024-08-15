import itertools
import signal
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
from shapely import STRtree
from shapely.geometry import box

# from scipy.spatial.distance import cityblock
from tqdm.auto import tqdm

import graphx as nx
from input import Flip_Flop, Inst, Net, PhysicalPin, Setting, VisualizeOptions, read_file, visualize
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
        self.setting = read_file(file_path)
        print("File read")
        self.G = self.build_dependency_graph(self.setting)
        self.pin_mapper = self.build_pin_mapper()
        self.G_clk = self.build_clock_graph(self.setting)
        # self.calculate_undefined_slack()
        print("Pin mapper created")
        self.flip_flop_query = self.build_ffs_query()
        self.ensure_ff_slacks()
        self.pin_mapping_info = []
        print("MBFFG created")

    def build_ffs_query(self):
        visited_ff_names = set()
        ffs = {}
        for node, data in self.G.nodes(data="pin"):
            if data.is_ff and data.inst.name not in visited_ff_names:
                visited_ff_names.add(data.inst.name)
                ffs[data.inst.name] = data.inst
        return ffs

    def ensure_ff_slacks(self):
        for inst in self.setting.instances:
            if inst.is_ff:
                assert [self.get_pin(dpin).slack for dpin in inst.dpins].count(
                    None
                ) == 0, f"FF {inst.name} has None slack"

    def build_pin_mapper(self):
        pin_mapper = {}
        for node, data in self.G.nodes(data="pin"):
            pin_mapper[node] = data
        return pin_mapper

    def build_dependency_graph(self, setting: Setting):
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
                G.add_edge(output_pin.full_name, pin.full_name)
        return G

    def build_clock_graph(self, setting):
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
        return G_clk

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

    def get_origin_pin(self, pin_name) -> PhysicalPin:
        if self.get_pin(pin_name).is_ff:
            while (p := self.pin_mapper[pin_name]).slack is None:
                # p = self.pin_mapper[p.full_name].inst
                pin_name = self.pin_mapper[p.full_name].full_name
        else:
            p = self.pin_mapper[pin_name]
        return p

    def get_inst(self, pin_name):
        return self.G.nodes[pin_name]["pin"].inst

    def get_insts(self, pin_names):
        return [self.G.nodes[pin_name]["pin"].inst for pin_name in pin_names]

    def get_pin(self, pin_name) -> PhysicalPin:
        return self.G.nodes[pin_name]["pin"]

    # def get_prev_ffs_path(self, node_name, temp_path=None, full_path=None):
    #     assert (temp_path is not None and full_path is not None) or (
    #         temp_path is None and full_path is None
    #     )
    #     if temp_path is None:
    #         temp_path = []
    #         full_path = []

    #     temp_path = temp_path + [node_name]
    #     node_pin = self.get_pin(node_name)
    #     if node_pin.is_gt:
    #         if node_pin.is_out:
    #             neightbors = [n for n in self.get_inst(node_name).inpins]
    #         elif node_pin.is_in:
    #             neightbors = []
    #             for n in self.G.neighbors(node_name):
    #                 if self.get_inst(n).name != node_pin.inst.name:
    #                     neightbors.append(n)
    #             neightbors = [
    #                 n
    #                 for n in self.G.neighbors(node_name)
    #                 if self.get_inst(n).name != node_pin.inst.name
    #             ]
    #         else:
    #             raise ValueError(f"Unknown gate type {node_pin}")
    #     else:
    #         neightbors = list(self.G.neighbors(node_name))

    #     for neighbor in neightbors:
    #         neighbor_pin = self.get_pin(neighbor)
    #         if neighbor_pin.is_io:
    #             # full_path.append((temp_path + [neighbor])[::-1])
    #             pass
    #         elif neighbor_pin.is_ff:
    #             if neighbor_pin.is_q:
    #                 full_path.append((temp_path + [neighbor])[::-1])
    #         else:
    #             self.get_prev_ffs_path(neighbor, temp_path, full_path)
    #     return full_path

    # @cache
    # def c_get_prev_ffs_path(self, node_name, parent_neighbors=None):
    #     neightbors_ori = set(self.G.neighbors(node_name))
    #     neightbors = neightbors_ori - (parent_neighbors if parent_neighbors else set())
    #     neightbors = frozenset(neightbors)
    #     result = []
    #     for neighbor in neightbors:
    #         neighbor_pin = self.get_pin(neighbor)
    #         if neighbor_pin.is_io:
    #             pass
    #         elif neighbor_pin.is_ff:
    #             if neighbor_pin.is_q:
    #                 result.append((neighbor, node_name))
    #         else:
    #             result.extend(self.c_get_prev_ffs_path(neighbor, neightbors))
    #     return list(set(result))

    @cache
    def prev_ffs_cache(self):
        return self.G.build_outgoing_map(Q_TAG, D_TAG)

    def get_prev_ffs(self, node_name: str):
        return self.prev_ffs_cache()[node_name]

    @cache
    def prev_pin_cache(self):
        return self.G.get_all_incomings(D_TAG)

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

        # def get_prev_inst_pin(self, node_name):
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

        # def get_fol_inst_pins(self, node_name):
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
        for i, (qpin, pff) in enumerate(prev_ffs):
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

    def merge_ff(self, insts: str | list, lib: str, libid):
        insts = self.get_ffs(insts)
        G = self.G
        pin_mapper = self.pin_mapper
        assert lib in self.setting.library, f"Library {lib} not found"
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
                    dpin_fullname = new_inst.dpins[dindex]
                    for neighbor in G.outgoings(pin.full_name):
                        G.add_edge(dpin_fullname, neighbor)
                    for neighbor in G.incomings(pin.full_name):
                        G.add_edge(neighbor, dpin_fullname)
                    dindex += 1
                    pin_mapper[dpin_fullname] = pin
                    self.pin_mapping_info.append((pin.full_name, dpin_fullname))
                elif pin.is_q:
                    # qpin_fullname = f"{new_pin.inst_name}/{qpin_name}"
                    qpin_fullname = new_inst.qpins[qindex]
                    for neighbor in G.outgoings(pin.full_name):
                        G.add_edge(qpin_fullname, neighbor)
                    for neighbor in G.incomings(pin.full_name):
                        G.add_edge(neighbor, qpin_fullname)
                    qindex += 1
                    pin_mapper[qpin_fullname] = pin
                    self.pin_mapping_info.append((pin.full_name, qpin_fullname))
                else:
                    new_pin_name = new_inst.pins_query[pin.name].full_name
                    for neighbor in G.outgoings(pin.full_name):
                        G.add_edge(new_pin_name, neighbor)
                    pin_mapper[new_pin_name] = pin
                    self.pin_mapping_info.append((pin.full_name, new_pin.full_name))
                G.remove_node(pin.full_name)
            del self.flip_flop_query[inst.name]
        new_pos = np.mean([x.pos for x in insts], axis=0)
        new_inst.moveto(new_pos)
        new_inst.libid = libid
        self.flip_flop_query[new_inst.name] = new_inst
        return new_inst

    def get_ff(self, ff_name) -> Inst:
        return self.flip_flop_query[ff_name]

    def get_ffs(self, ff_names: str = None) -> list[Inst]:
        if ff_names is None:
            return list(self.flip_flop_query.values())
        else:
            if isinstance(ff_names, str):
                ff_names = ff_names.split(",")
            try:
                return [self.get_ff(ff_name) for ff_name in ff_names]
            except:
                assert False, f"FFs {ff_names} not found"

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
            total_tns += -sum(slacks)
            total_power += ff.lib.power
            total_area += ff.lib.area
        print("Scoring done")
        # print(self.setting.alpha * total_tns)

        return (
            self.setting.alpha * total_tns
            + self.setting.beta * total_power
            + self.setting.gamma * total_area
        )

    def original_pin_distance(self, node1, node2):
        return cityblock(self.get_origin_pin(node1).pos, self.get_origin_pin(node2).pos)

    def current_pin_distance(self, node1, node2):
        return cityblock(self.get_pin(node1).pos, self.get_pin(node2).pos)

    @static_vars(graph_num=1)
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
                file_name=f"output{MBFFG.transfer_graph_to_setting.graph_num}.{extension}",
                resolution=None if extension == "html" else 10000,
            )
            MBFFG.transfer_graph_to_setting.graph_num += 1

    def print_graph(G):
        for node, data in G.nodes(data="pin"):
            print(node, list(G.neighbors(node)))

    def reset_cache(self):
        self.prev_ffs_cache.cache_clear()
        self.prev_pin_cache.cache_clear()

    def optimize(self, global_optimize=True):
        # self.reset_cache()

        def cityblock_variable(model, v1, v2, bias, weight, intercept):
            # delta_x, delta_y = model.addVar(lb=-GRB.INFINITY), model.addVar(lb=-GRB.INFINITY)
            abs_delta_x, abs_delta_y = model.addVar(), model.addVar()
            # model.addLConstr(delta_x == v1[0] - v2[0])
            # model.addLConstr(delta_y == v1[1] - v2[1])
            delta_x = v1[0] - v2[0]
            delta_y = v1[1] - v2[1]
            model.addLConstr(abs_delta_x >= delta_x)
            model.addLConstr(abs_delta_x >= -delta_x)
            model.addLConstr(abs_delta_y >= delta_y)
            model.addLConstr(abs_delta_y >= -delta_y)
            cityblock_distance = model.addVar(lb=-GRB.INFINITY)
            model.addLConstr(
                cityblock_distance == weight * (abs_delta_x + abs_delta_y + bias) + intercept
            )
            return cityblock_distance

        print("Optimizing...")
        # k = [
        #     ff
        #     for ff in self.get_ffs()
        #     if any([self.get_origin_pin(curpin).slack < 0 for curpin in ff.dpins])
        # ]
        with gp.Env(empty=True) as env:
            env.setParam("LogToConsole", 1)
            env.start()

            def solve(optimize_ffs):
                with gp.Model(env=env) as model:
                    for ff in optimize_ffs:
                        ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                            name=ff.name + "1"
                        )
                    model.setParam("OutputFlag", 0)

                    # model.setParam(GRB.Param.Presolve, 2)
                    # if global_optimize:
                    # else:
                    #     # optimize_ffs_names = [ff.name for ff in optimize_ffs]
                    #     # print(optimize_ffs_names)
                    #     # for ff in self.get_ffs():
                    #     #     if ff.name in optimize_ffs_names:
                    #     #         ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                    #     #             name=ff.name + "1"
                    #     #         )
                    #     # else:
                    #     #     ff_vars[ff.name] = ff.pos
                    #     for ff in optimize_ffs:
                    #         ff_vars[ff.name] = model.addVar(name=ff.name + "0"), model.addVar(
                    #             name=ff.name + "1"
                    #         )

                    # dis2ori_locations = []
                    negative_slack_vars = []
                    for ff in optimize_ffs:
                        for curpin in ff.dpins:
                            ori_slack = self.get_origin_pin(curpin).slack
                            prev_pin = self.get_prev_pin(curpin)
                            prev_pin_displacement_delay = 0
                            if prev_pin:
                                current_pin = self.get_pin(curpin)
                                current_pin_pos = [
                                    a + b
                                    for a, b in zip(
                                        ff_vars[current_pin.inst.name], current_pin.rel_pos
                                    )
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
                                # prev_pin_displacement_delay = model.addVar()

                            displacement_distances = []
                            prev_ffs = self.get_prev_ffs(curpin)
                            for qpin, pff in prev_ffs:
                                pff_pin = self.get_pin(pff)
                                qpin_pin = self.get_pin(qpin)
                                pff_pos = [
                                    a + b
                                    for a, b in zip(ff_vars[pff_pin.inst.name], pff_pin.rel_pos)
                                ]
                                if qpin_pin.is_ff:
                                    qpin_pos = [
                                        a + b
                                        for a, b in zip(
                                            ff_vars[qpin_pin.inst.name], qpin_pin.rel_pos
                                        )
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
                                # distance_var = model.addVar()
                                displacement_distances.append(distance_var)
                            min_displacement_distance = 0
                            if len(displacement_distances) > 0:
                                min_displacement_distance = model.addVar(lb=-GRB.INFINITY)
                                model.addConstr(
                                    min_displacement_distance == gp.min_(displacement_distances)
                                )

                            slack_var = model.addVar(name=curpin, lb=-GRB.INFINITY)
                            model.addConstr(
                                slack_var
                                == ori_slack
                                - (prev_pin_displacement_delay + min_displacement_distance)
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
                    for ff in optimize_ffs:
                        name = ff.name
                        new_pos = (ff_vars[name][0].X, ff_vars[name][1].X)
                        self.get_ff(name).moveto(new_pos)
                    # for name, ff_var in ff_vars.items():
                    #     if isinstance(ff_var[0], float):
                    #         self.get_ffs(name)[0].moveto((ff_var[0], ff_var[1]))
                    #     else:
                    #         self.get_ffs(name)[0].moveto((ff_var[0].X, ff_var[1].X))
                    #     ff_vars[name] = self.get_ff(name).pos

            ff_vars = self.get_static_vars()
            if not global_optimize:
                pin_list = self.get_end_ffs()
                # pin_name = pin_list[0]
                ff_paths = set()
                ffs_calculated = set()
                ff_path_all = [self.get_ff_path(pin_name) for pin_name in pin_list]
                ff_path_all.sort(key=lambda x: len(x), reverse=True)
                for ff_path in tqdm(ff_path_all):
                    ff_paths.update(ff_path)
                    # if len(ff_paths) < 2000:
                    #     ff_paths.update(ff_path)
                    #     continue
                    solve([self.get_ff(pin_name) for pin_name in (ff_paths - ffs_calculated)])
                    for name in ff_paths:
                        ff_vars[name] = self.get_ff(name).pos
                    ffs_calculated.update(ff_paths)
                    ff_paths.clear()
            else:
                solve(self.get_ffs())
        # self.legalization_rust(ff_vars)
        # self.legalization_check()

    def get_static_vars(self):
        return {ff.name: ff.pos for ff in self.get_ffs()}

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

    def legalization_rust(self):
        assert all([x.libid is not None for x in self.get_ffs()]), "FF idx is None"
        ff_vars = self.get_static_vars()
        print("Legalizing...")
        aabbs = []
        for placement_row in self.setting.placement_rows:
            for i in range(placement_row.num_cols):
                x, y = placement_row.x + i * placement_row.width, placement_row.y
                aabbs.append(((x, y), (x + placement_row.width, y + placement_row.height)))

        barriers = [gate.bbox_corner for gate in self.get_gates()]
        ff_names = list(ff_vars.keys())
        ff_names.sort(key=lambda x: (self.get_ff(x).libid))
        candidates = [(self.get_ff(x).libid, self.get_ff(x).bbox_corner) for x in ff_names]
        borders = self.setting.die_size.bbox_corner
        result, size = rustlib.legalize(aabbs, barriers, candidates, borders)
        for i in range(size):
            name = ff_names[i]
            ff = self.get_ff(name)
            ff.moveto(result[i])
        # if size != len(candidates):
        #     self.cvdraw()
        assert size == len(candidates), f"Size not match {size} {len(candidates)}"

    def legalization_check(self):
        boxes = [box(*gate.bbox) for gate in self.get_gates()]
        tree = STRtree(boxes)
        for ff in self.get_ffs():
            bbox = ff.bbox
            target = box(*bbox).buffer(-0.01)
            indices = tree.query(target)
            for index in indices:
                if boxes[index].intersects(target):
                    print(f"FF {ff.name} intersects with {index}")
                    print(boxes[index].bounds, target.bounds)
                    exit()
            border = np.array(self.setting.die_size.bbox_corner).flatten()
            if (
                bbox[0] < border[0]
                or bbox[1] < border[1]
                or bbox[2] > border[2]
                or bbox[3] > border[3]
            ):
                print(f"FF {ff.name} out of border")
                print(border, bbox)
                exit()

    def output(self, path):
        with open(path, "w") as file:
            file.write(f"CellInst {len(self.get_ffs())}\n")
            for ff in self.get_ffs():
                file.write(f"Inst {ff.name} {ff.lib.name} {ff.pos[0]} {ff.pos[1]}\n")
            for f, t in self.pin_mapping_info:
                file.write(f"{f} map {t}\n")

    @static_vars(graph_num=1)
    def cvdraw(self, filename=None):
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        BLACK = (0, 0, 0)
        img_width = self.setting.die_size.xUpperRight
        img_height = self.setting.die_size.yUpperRight
        ratio = 5000 / max(img_width, img_height)
        img_width, img_height = int(img_width * ratio), int(img_height * ratio)
        img = np.ones((img_height, img_width, 3), np.uint8) * 255
        border_width = 20
        cv2.line(img, (0, 0), (0, img_height), RED, border_width)
        cv2.line(img, (0, 0), (img_width, 0), RED, border_width)
        cv2.line(img, (img_width - 1, 0), (img_width - 1, img_height), RED, border_width)
        cv2.line(img, (0, img_height - 1), (img_width, img_height - 1), RED, border_width)
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
        if not filename:
            file_name = f"output{MBFFG.cvdraw.graph_num}.png"
            MBFFG.cvdraw.graph_num += 1
        else:
            file_name = filename + ".png"
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

    def get_end_ffs(self):
        return [
            pin_name
            for pin_name, connections in self.G.build_incoming_map(D_TAG, Q_TAG).items()
            if len(connections) == 0
        ]

    def get_ff_path(self, end_pin_name) -> set[str]:
        # outgoing_map = self.G.build_outgoing_map(Q_TAG, D_TAG)
        waiting = [end_pin_name]
        inst_list = set()
        while waiting:
            inst_name = self.get_pin(waiting[0]).inst.name
            inst = self.get_ff(inst_name)
            if inst_name not in inst_list:
                inst_list.add(inst_name)
                for d in inst.dpins:
                    connected_q = self.get_prev_ffs(d)
                    if connected_q:
                        waiting.append(connected_q[0][1])
            waiting.pop(0)
        return inst_list


# def get_pin_name(node_name):
#     return node_name.split("/")[1]
