import copy
import itertools
import time
from collections import defaultdict
from functools import cache, cached_property, partial
from pprint import pprint

import cv2
import gurobipy as gp

# import networkx as nx
import numpy as np
import prettytable as pt
import rtree
import rustlib
from gurobipy import GRB
from prettytable import PrettyTable
from scipy.spatial import KDTree
from shapely import STRtree
from shapely.geometry import box
from sortedcontainers import SortedDict

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
        self.flip_flop_query = self.build_ffs_query()
        self.build_clock_graph(self.setting)
        # self.ensure_ff_slacks()
        self.pin_mapping_info = []
        self.alter_ffs: list[str] = []
        self.initial_ff_names = list(self.flip_flop_query.keys())
        print("MBFFG created")

    def build_ffs_query(self) -> dict[str, Inst]:
        ffs = {}
        for node, data in self.G.nodes(data="pin"):
            if data.is_ff:
                ffs.setdefault(data.inst.name, data.inst)
        return ffs

    def ensure_ff_slacks(self):
        for inst in self.setting.instances:
            if inst.is_ff:
                assert [self.get_pin(dpin).slack for dpin in inst.dpins].count(
                    None
                ) == 0, f"FF {inst.name} has None slack"

    def build_pin_mapper(self, G):
        inst_copy = {}
        for node, data in G.nodes(data="pin"):
            if data.is_ff:
                data_bk = copy.copy(data)
                data_bk.inst = inst_copy.setdefault(data_bk.inst_name, copy.copy(data_bk.inst))
                G.nodes[node]["pin"] = data_bk
        for inst in inst_copy.values():
            inst.assign_pins([G.nodes[x.full_name]["pin"] for x in inst.pins])

    def build_dependency_graph(self, setting: Setting):
        G = nx.DiGraph()
        for inst in setting.instances:
            # if inst.is_gt:
            #     in_pins = [pin.full_name for pin in inst.pins if pin.is_in]
            #     out_pins = [pin.full_name for pin in inst.pins if pin.is_out]
            #     G.add_edges_from(itertools.product(in_pins, out_pins))
            # elif inst.is_ff:
            #     d_pins = [pin.full_name for pin in inst.pins if pin.is_d]
            #     q_pins = [pin.full_name for pin in inst.pins if pin.is_q]
            #     G.add_edges_from(itertools.product(d_pins, q_pins))
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
        inst_copy = {}
        for node, data in G.nodes(data="pin"):
            if data.is_ff:
                data_bk = copy.copy(data)
                data_bk.inst = inst_copy.setdefault(data_bk.inst_name, copy.copy(data_bk.inst))
                G.nodes[node]["pin"] = data_bk
        for inst in inst_copy.values():
            inst.assign_pins([G.nodes[x.full_name]["pin"] for x in inst.pins])
        return G

    def build_clock_graph(self, setting: Setting) -> tuple[dict[str, int], list[list[str]]]:
        inst_clk_nets = defaultdict(list)
        clk_nets = []
        for net in setting.nets:
            pins = [pin for pin in net.pins if pin.is_clk]
            if len(pins) == 0:
                continue
            clk_nets.append([pin.inst.name for pin in pins])
            for pin in pins:
                inst_clk_nets[pin.inst.name].append(len(clk_nets) - 1)
        for inst_name, nets in inst_clk_nets.items():
            assert len(nets) == 1, f"Multiple clock nets for {inst_name}"
            self.get_ff(inst_name).clk_neighbor = clk_nets[nets[0]]
            # inst_clk_nets[inst_name] = nets[0]

    def get_origin_pin(self, pin_name) -> PhysicalPin:
        pin: PhysicalPin = self.get_pin(pin_name)
        if pin.is_io:
            return pin
        else:
            return self.setting.inst_query[pin.ori_inst_name()].pins_query[pin.ori_name()]

    # def get_origin_inst(self, name, *, pin=True) -> Inst:
    #     if pin:
    #         return self.get_origin_inst(self.G.nodes[name]["pin"].inst_name, pin=False)
    #     else:
    #         return self.setting.inst_query[name]

    # def get_inst(self, name, *, pin=True) -> Inst:
    #     if pin:
    #         return self.G.nodes[name]["pin"].inst
    #     else:
    #         return self.get_inst(self.setting.inst_query[name].pins[0].full_name)

    def get_pin(self, pin_name) -> PhysicalPin:
        return self.G.nodes[pin_name]["pin"]

    # def get_inst(self, inst_name) -> Inst:
    #     return self.get_pin(self.setting.inst_query[inst_name].pins[0].full_name).inst

    @cache
    def prev_ffs_cache(self):
        return self.G.build_incoming_map(Q_TAG, D_TAG)

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
        return (
            self.get_origin_pin(node_name).inst.qpin_delay - self.get_pin(node_name).inst.qpin_delay
        )

    def merge_ff(self, inst_names: str | list, lib: str, libid: int):
        insts = self.get_ffs(inst_names)
        G = self.G
        assert lib in self.setting.library, f"Library {lib} not found"
        assert (
            sum([inst.lib.bits for inst in insts]) == self.setting.library[lib].bits
        ), f"FFs not match target {self.setting.library[lib].bits} bits lib, try to merge {sum([inst.lib.bits for inst in insts])} bits, from {insts} to {lib}"
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
            self.alter_ffs.append(inst.name)
            for pin in inst.pins:
                if pin.is_d:
                    dpin_fullname = new_inst.dpins[dindex]
                    dpin = self.get_pin(dpin_fullname)
                    dpin.origin_inst_name = pin.ori_inst_name()
                    dpin.origin_name = pin.ori_name()
                    # for neighbor in G.outgoings(pin.full_name):
                    #     G.add_edge(dpin_fullname, neighbor)
                    for neighbor in G.incomings(pin.full_name):
                        G.add_edge(neighbor, dpin_fullname)
                    dindex += 1
                    self.pin_mapping_info.append((pin.full_name, dpin_fullname))
                elif pin.is_q:
                    qpin_fullname = new_inst.qpins[qindex]
                    self.get_pin(qpin_fullname).origin_inst_name = pin.ori_inst_name()
                    self.get_pin(qpin_fullname).origin_name = pin.ori_name()
                    for neighbor in G.outgoings(pin.full_name):
                        G.add_edge(qpin_fullname, neighbor)
                    # for neighbor in G.incomings(pin.full_name):
                    #     G.add_edge(neighbor, qpin_fullname)
                    qindex += 1
                    self.pin_mapping_info.append((pin.full_name, qpin_fullname))
                elif pin.is_clk:
                    new_pin_name = new_inst.pins_query[pin.name].full_name
                    self.get_pin(new_pin_name).origin_inst_name = pin.ori_inst_name()
                    self.get_pin(new_pin_name).origin_name = pin.ori_name()
                    for neighbor in G.incomings(pin.full_name):
                        G.add_edge(new_pin_name, neighbor)
                    self.pin_mapping_info.append((pin.full_name, new_pin.full_name))
                G.remove_node(pin.full_name)
            del self.flip_flop_query[inst.name]
        new_pos = np.mean([x.pos for x in insts], axis=0)
        new_inst.moveto(new_pos)
        new_inst.libid = libid
        self.flip_flop_query[new_inst.name] = new_inst
        return new_inst

    def demerge(self, inst_name, lib_name):
        G = self.G
        inst = self.get_ff(inst_name)
        lib = self.setting.library[lib_name]
        assert inst.lib.bits % lib.bits == 0, f"FF {inst_name} bits not match lib {lib_name}"
        self.alter_ffs.append(inst.name)
        new_insts: list[Inst] = []
        for i in range(inst.bits // lib.bits):
            new_inst = self.setting.get_new_instance(lib_name)
            new_inst.name += "_" + str(i) + "_" + inst_name
            new_inst.moveto((inst.x + (new_inst.width + 1) * i, inst.y))
            for pin in new_inst.pins:
                G.add_node(pin.full_name, pin=pin)
                if pin.is_q:
                    G.add_tag(pin.full_name, Q_TAG)
                elif pin.is_d:
                    G.add_tag(pin.full_name, D_TAG)
            self.flip_flop_query[new_inst.name] = new_inst
            new_insts.append(new_inst)
        del self.flip_flop_query[inst_name]
        inst_dpins = inst.dpins
        inst_qpins = inst.qpins
        inst_clkpin = self.get_pin(inst.clkpin)
        cidx = 0
        iidx = 0
        while cidx < inst.bits:
            new_inst = new_insts[iidx]
            for i in range(lib.bits):
                inst_dpin = self.get_pin(inst_dpins[cidx + i])
                new_dpin_fullname = new_inst.dpins[i]
                new_inst_dpin = self.get_pin(new_dpin_fullname)
                new_inst_dpin.origin_inst_name = inst_dpin.ori_inst_name()
                new_inst_dpin.origin_name = inst_dpin.ori_name()
                for neighbor in G.incomings(inst_dpin.full_name):
                    G.add_edge(neighbor, new_dpin_fullname)
                self.pin_mapping_info.append((inst_dpin.full_name, new_dpin_fullname))

                inst_qpin = self.get_pin(inst_qpins[cidx + i])
                new_qpin_fullname = new_inst.qpins[i]
                new_inst_qpin = self.get_pin(new_qpin_fullname)
                new_inst_qpin.origin_inst_name = inst_qpin.ori_inst_name()
                new_inst_qpin.origin_name = inst_qpin.ori_name()
                for neighbor in G.outgoings(inst_qpin.full_name):
                    G.add_edge(new_qpin_fullname, neighbor)
                self.pin_mapping_info.append((inst_qpin.full_name, new_qpin_fullname))
                G.remove_node(inst_dpin.full_name)
                G.remove_node(inst_qpin.full_name)
            new_inst_clkpin = self.get_pin(new_inst.clkpin)
            for neighbor in G.incomings(inst_clkpin.full_name):
                G.add_edge(neighbor, new_inst_clkpin.full_name)
            new_inst_clkpin.origin_inst_name = inst_clkpin.ori_inst_name()
            new_inst_clkpin.origin_name = inst_clkpin.ori_name()
            inst.clk_neighbor.append(new_inst.name)
            new_inst.clk_neighbor = inst.clk_neighbor
            iidx += 1
            cidx += lib.bits
        G.remove_node(inst_clkpin.full_name)

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

    def utilization_score(self):
        num = 0
        bin_width = self.setting.bin_width
        bin_height = self.setting.bin_height
        bin_max_util = self.setting.bin_max_util
        die_size = self.setting.die_size
        insts = [box(*gate.bbox) for gate in self.get_gates()] + [
            box(*ff.bbox) for ff in self.get_ffs()
        ]
        tree = STRtree(insts)
        anchor = [0, 0]
        while anchor[1] < die_size.yUpperRight:
            if anchor[0] < die_size.xUpperRight:
                area = 0
                query_box = box(anchor[0], anchor[1], anchor[0] + bin_width, anchor[1] + bin_height)
                overlap = tree.query(query_box)
                true_overlap = [x for x in overlap if insts[x].intersects(query_box)]
                if len(true_overlap) > 0:
                    # print("Overlap", true_overlap)
                    for idx in overlap:
                        # print(query_box.bounds, insts[idx].bounds)
                        # print(insts[idx].intersection(query_box).area)
                        area += insts[idx].intersection(query_box).area
                if area > bin_max_util:
                    num += 1
            else:
                anchor[0] = 0
                anchor[1] += bin_height
                continue
            anchor[0] += bin_width
        return num

    def timing_slack(self, node_name):
        node_pin = self.get_pin(node_name)
        if node_pin.is_in or node_pin.is_gt or node_pin.is_q:
            return 0
        # print(node_name)
        # print(self.get_pin(node_name).ori_pin_name())
        # exit()
        assert self.get_origin_pin(node_name).slack is not None, f"No slack for {node_name}"
        # print(self.get_pin(node_name))
        # exit()
        self_displacement_delay = 0
        prev_pin = self.get_prev_pin(node_name)
        if prev_pin:
            self_displacement_delay = self.original_pin_distance(
                prev_pin, node_name
            ) - self.current_pin_distance(prev_pin, node_name)

        prev_ffs = self.get_prev_ffs(node_name)
        assert len(prev_ffs) <= 1, f"Multiple previous FFs for {node_name}, {prev_ffs}"
        prev_ffs_qpin_displacement_delay = 0
        prev_ffs_qpin_delay = 0
        if len(prev_ffs) == 1:
            pff, qpin = prev_ffs[0]
            prev_ffs_qpin_displacement_delay = self.original_pin_distance(
                pff, qpin
            ) - self.current_pin_distance(pff, qpin)
            prev_ffs_qpin_delay = self.qpin_delay_loss(qpin)

        total_delay = (
            prev_ffs_qpin_delay
            + self.get_origin_pin(node_name).slack
            + (self_displacement_delay + prev_ffs_qpin_displacement_delay)
            * self.setting.displacement_delay
        )
        return total_delay

    def scoring(self):
        print("Scoring...")
        total_tns = 0
        total_power = 0
        total_area = 0
        statistics = NestedDict()
        for ff in tqdm(self.get_ffs()):
            slacks = [min(self.timing_slack(dpin), 0) for dpin in ff.dpins]
            total_tns += -sum(slacks)
            total_power += ff.lib.power
            total_area += ff.lib.area
            statistics["ff"][ff.bits] = statistics["ff"].get(ff.bits, 0) + 1
        statistics["total_gate"] = len(self.get_gates())
        statistics["total_ff"] = len(self.get_ffs())
        tns_score = self.setting.alpha * total_tns
        power_score = self.setting.beta * total_power
        area_score = self.setting.gamma * total_area
        utilization_score = self.setting.lambde * self.utilization_score()
        total_score = tns_score + power_score + area_score + utilization_score
        tns_ratio = round(tns_score / total_score * 100, 2)
        power_ratio = round(power_score / total_score * 100, 2)
        area_ratio = round(area_score / total_score * 100, 2)
        utilization_ratio = round(utilization_score / total_score * 100, 2)
        statistics["score"]["tns"] = tns_score
        statistics["score"]["power"] = power_score
        statistics["score"]["area"] = area_score
        statistics["score"]["utilization"] = utilization_score
        statistics["ratio"]["tns"] = tns_ratio
        statistics["ratio"]["power"] = power_ratio
        statistics["ratio"]["area"] = area_ratio
        statistics["ratio"]["utilization"] = utilization_ratio
        statistics["score"]["total"] = total_score
        # print("Scoring done")
        return total_score, statistics

    def show_statistics(self, stat1, stat2):
        table = PrettyTable()
        table.field_names = ["", "Score", "Gap", "Gap Ratio", "Weight", "Improvement"]
        stat1_score = stat1["score"]
        stat2_score = stat2["score"]
        stat2["ratio"]["total"] = 100
        for name, key in zip(
            ["TNS", "Power", "Area", "Util", "Total"],
            ["tns", "power", "area", "utilization", "total"],
        ):
            table.add_row(
                [
                    name,
                    f"{stat1_score[key]} -> {stat2_score[key]}",
                    (diff := stat2_score[key] - stat1_score[key]),
                    str((diff) / (stat2_score["total"] - stat1_score["total"] + 1e-5) * 100) + "%",
                    f"{stat2["ratio"][key]}%",
                    (
                        f"{round(diff / stat1_score[key] * 100, 2)}%"
                        if stat1_score[key] != 0
                        else float("inf") if diff != 0 else "0.0%"
                    ),
                ],
                divider=True if name == "Util" else False,
            )
        table.float_format = ".2"
        # table.align["Weight"] = "r"
        table.align["Improvement"] = "r"

        table.set_style(pt.SINGLE_BORDER)
        print(table)

        table = PrettyTable()
        table.set_style(pt.SINGLE_BORDER)
        possible_bits = set(stat1["ff"].keys()) | set(stat2["ff"].keys())
        table.field_names = ["FFs", "Number of FFs"]
        for k in possible_bits:
            table.add_row([f"{k} bits", f"{stat1['ff'].get(k,0)} -> {stat2['ff'].get(k, 0)}"])
        print(table)

    def original_pin_distance(self, node1, node2):
        return cityblock(self.get_origin_pin(node1).pos, self.get_origin_pin(node2).pos)

    def current_pin_distance(self, node1, node2):
        return cityblock(self.get_pin(node1).pos, self.get_pin(node2).pos)

    @static_vars(graph_num=1)
    def transfer_graph_to_setting(self, options, visualized=True, show_distance=False):
        if self.G.size > 1000:
            return
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
            net = Net("n", num_pins=2)
            net.pins = [self.get_pin(node1), self.get_pin(node2)]
            if show_distance:
                net.metadata = cityblock(net.pins[0].pos, net.pins[1].pos)
            setting.nets.append(net)
        if visualized:
            visualize(
                setting,
                options,
                file_name=f"output/output{MBFFG.transfer_graph_to_setting.graph_num}.{extension}",
                resolution=None if extension == "html" else 10000,
            )
            MBFFG.transfer_graph_to_setting.graph_num += 1

    # def print_graph(G):
    #     for node, data in G.nodes(data="pin"):
    #         print(node, list(G.neighbors(node)))

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
        # for x in self.get_ffs():
        #     assert x.libid is not None, f"FF '{x.name}' idx is None"
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
        border = np.array(self.setting.die_size.bbox_corner).flatten()
        for ff in self.get_ffs():
            bbox = ff.bbox
            target = box(*bbox).buffer(-0.01)
            indices = tree.query(target)
            for index in indices:
                if boxes[index].intersects(target):
                    print(f"FF {ff.name} intersects with {index}")
                    print(boxes[index].bounds, target.bounds)
                    exit()
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
            # for f, t in self.pin_mapping_info:
            #     file.write(f"{f} map {t}\n")
            # not_mapped_inst_names = set(self.initial_ff_names) - set(self.alter_ffs)
            # for inst_name in not_mapped_inst_names:
            #     for pin in self.get_ff(inst_name).pins:
            #         file.write(f"{pin.full_name} map {pin.full_name}\n")
            for ff in self.get_ffs():
                prev_inst = set()
                for pin in ff.pins:
                    if not pin.is_clk:
                        file.write(f"{pin.ori_full_name()} map {pin.full_name}\n")
                        prev_inst.add(pin.ori_inst_name())
                for prev_inst_name in prev_inst:
                    file.write(f"{self.setting.inst_query[prev_inst_name].clkpin} map {ff.clkpin}\n")

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
        # library_sorted = sorted(
        #     self.get_library().values(),
        #     key=lambda x: (x.power * self.setting.beta + x.area * self.setting.gamma) / x.bits,
        # )
        library_seg_values = {
            x.name: (x, ((x.power * self.setting.beta + x.area * self.setting.gamma) / x.bits))
            for x in self.get_library().values()
        }
        library_seg_best_values = defaultdict(lambda: float("inf"))
        library_seg_best: dict[int, Flip_Flop] = {}
        for ff, score in library_seg_values.values():
            if library_seg_best_values[ff.bits] > score:
                library_seg_best[ff.bits] = ff
                library_seg_best_values[ff.bits] = score
        library_seg_best_order = sorted(
            library_seg_best.items(), key=lambda x: library_seg_values[x[1].name][1]
        )
        lib_order = [x[0] for x in library_seg_best_order]
        return library_seg_best, lib_order

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

    def demerge_ffs(self):
        optimal_library_segments, library_sizes = self.get_selected_library()
        min_lib_size = min(library_sizes)
        min_lib_name = optimal_library_segments[min_lib_size].name
        for ff in self.get_ffs():
            if ff.bits != min_lib_size and ff.bits % min_lib_size == 0:
                self.demerge(ff.name, min_lib_name)
        self.reset_cache()


# def get_pin_name(node_name):
#     return node_name.split("/")[1]
