import copy
import itertools
import time
from collections import defaultdict
from pprint import pprint

import gurobipy as gp
import networkx as nx
import numpy as np
import rtree
import shapely.ops as ops
from gurobipy import GRB
from scipy.spatial.distance import cityblock
from shapely.geometry import Point

from input import Inst, Net, PhysicalPin, read_file, visualize

print_tmp = print
print = lambda *args: (
    print_tmp(*args) if len(args) > 1 else pprint(args[0]) if args else print_tmp()
)


class MBFFG:
    def __init__(self, file_path):
        setting = read_file(file_path)
        G = nx.Graph()
        for inst in setting.instances:
            if not inst.is_ff:
                G.add_edges_from(itertools.combinations([pin.full_name for pin in inst.pins], 2))
            for pin in inst.pins:
                G.add_node(pin.full_name, pin=pin)
        for input in setting.inputs:
            for pin in input.pins:
                G.add_node(pin.full_name, pin=pin)
        for output in setting.outputs:
            for pin in output.pins:
                G.add_node(pin.full_name, pin=pin)
        for net in setting.nets:
            output_pin = net.pins[0]
            for pin in net.pins[1:]:
                G.add_edge(output_pin.full_name, pin.full_name)
        self.G = G
        self.setting = setting
        self.graph_num = 1
        self.calculate_undefined_slack()
        pin_mapper = {}
        for node, data in G.nodes(data="pin"):
            pin_mapper[node] = data
        self.pin_mapper = pin_mapper
        self.G = copy.deepcopy(G)
        ff_filter = set()
        ffs = {}
        for node, data in self.G.nodes(data="pin"):
            if data.is_ff and data.inst.name not in ff_filter:
                ff_filter.add(data.inst.name)
                ffs[data.inst.name] = data.inst
        self.ffs = ffs
        self.new_ffs = []

    def calculate_undefined_slack(self):
        for input in self.setting.inputs:
            # print(input.name)
            for pin in self.G[input.name]:
                pin_node = self.get_pin(pin)
                if pin_node.name.startswith("d") and pin_node.slack is None:
                    pin_node.slack = 0
                    # print(f"Set slack for {pin_node.full_name} to 0")

        for inst in self.setting.instances:
            if inst.is_ff:
                for pin in inst.pins:
                    if pin.name.startswith("d") and pin.slack is None:
                        paths = self.get_prev_ffs_path(pin.full_name)
                        path_slacks = np.zeros(len(paths))
                        for i, path in enumerate(paths):
                            ff0, ffn = path[0], path[-1]
                            if len(path) == 2:
                                path_slacks[i] = self.get_inst(ff0).qpin_delay + (
                                    self.setting.displacement_delay
                                    * (cityblock(self.get_pin(ff0).pos, self.get_pin(ffn).pos))
                                )
                            else:
                                c0 = path[1]
                                cn = path[-2]
                                path_slacks[i] = self.get_inst(ff0).qpin_delay + (
                                    self.setting.displacement_delay
                                    * (
                                        cityblock(self.get_pin(ff0).pos, self.get_pin(c0).pos)
                                        + cityblock(self.get_pin(c0).pos, self.get_pin(cn).pos)
                                        + cityblock(self.get_pin(cn).pos, self.get_pin(ffn).pos)
                                    )
                                )
                        pin.slack = np.max(path_slacks)

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

    def get_prev_ffs_path(self, node_name, inst_counter=None, temp_path=None, full_path=None):
        assert (inst_counter is not None and temp_path is not None and full_path is not None) or (
            inst_counter is None and temp_path is None and full_path is None
        )
        if inst_counter is None:
            inst_counter = defaultdict(int)
            temp_path = []
            full_path = []

        temp_path = temp_path + [node_name]
        inst_counter[self.get_inst(node_name).name] += 1
        neightbors = [
            n
            for n in self.G.neighbors(node_name)
            if (not self.get_pin(n).is_gt) or (inst_counter[self.get_inst(n).name] <= 1)
        ]
        for neighbor in neightbors:
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io:
                # full_path.append((temp_path + [neighbor])[::-1])
                pass
            elif neighbor_pin.is_ff:
                if neighbor_pin.name.startswith("q"):
                    full_path.append((temp_path + [neighbor])[::-1])
            else:
                self.get_prev_ffs_path(neighbor, inst_counter, temp_path, full_path)
        return full_path

    def get_prev_ffs(self, node_name):
        return [(n[0], n[1]) for n in self.get_prev_ffs_path(node_name)]

    def get_prev_pin(self, node_name):
        prev_pins = []
        for neighbor in self.G.neighbors(node_name):
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
            neighbor_pin = self.get_pin(neighbor)
            if not neighbor_pin.is_io:
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
            neighbor_pin = self.get_pin(neighbor)
            if not neighbor_pin.is_io:
                res.append(neighbor)
        return res

    def get_ff_neighbor_inst_pin(self, node_name):
        inst = self.get_ffs(node_name)[0]
        res = []
        for dpin in inst.dpins:
            res.extend(self.get_prev_inst_pin(dpin))
        for qpin in inst.qpins:
            res.extend(self.get_fol_inst_pins(qpin))
        return res

    def min_distance_to_neightbor_inst_pin(self, node_name):
        inst = self.get_ffs(node_name)[0]
        min_distance = float("inf")
        min_before_distance = float("inf")
        pin = ""
        for dpin in inst.dpins:
            for pin in self.get_prev_inst_pin(dpin):
                min_distance = min(
                    min_distance, cityblock(self.get_pin(dpin).pos, self.get_pin(pin).pos)
                )
                if min_distance < min_before_distance:
                    min_before_distance = min_distance
                    pin = pin
        for qpin in inst.qpins:
            for pin in self.get_fol_inst_pins(qpin):
                min_distance = min(
                    min_distance, cityblock(self.get_pin(qpin).pos, self.get_pin(pin).pos)
                )
                if min_distance < min_before_distance:
                    min_before_distance = min_distance
                    pin = pin
        return min_distance, pin

    def timing_slack(self, node_name):
        if not self.get_pin(node_name).is_ff:
            return 0
        if not self.get_pin(node_name).name.startswith("d"):
            return 0
        assert self.get_pin(node_name).slack is not None, f"No slack for {node_name}"
        self_displacement_delay = 0
        prev_pin = self.get_prev_pin(node_name)
        if prev_pin:
            self_displacement_delay = (
                self.original_pin_distance(prev_pin, node_name)
                - self.current_pin_distance(prev_pin, node_name)
            ) * self.setting.displacement_delay
        prev_ffs = self.get_prev_ffs(node_name)
        prev_ffs_qpin_displacement_delay = [0]
        for pff, qpin in prev_ffs:
            prev_ffs_qpin_displacement_delay.append(
                self.get_origin_inst(pff).qpin_delay
                - self.get_inst(pff).qpin_delay
                + (self.original_pin_distance(pff, qpin) - self.current_pin_distance(pff, qpin))
                * self.setting.displacement_delay
            )
        # print(prev_ffs_qpin_displacement_delay)
        # print(prev_ffs)

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
            self.G.add_node(new_pin.full_name, pin=new_pin)

        dindex, qindex = 0, 0
        for inst in insts:
            for pin in inst.pins:
                if pin.name.startswith("d"):
                    dpin_name = f"d{dindex}" if len(insts) > 1 else "d"
                    dpin_fullname = f"{new_pin.inst_name}/{dpin_name}"
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(dpin_fullname, neightbor)
                    dindex += 1
                    new_inst.pins_query[dpin_name].slack = pin.slack
                    pin_mapper[dpin_fullname] = pin
                elif pin.name.startswith("q"):
                    qpin_name = f"q{qindex}" if len(insts) > 1 else "q"
                    qpin_fullname = f"{new_pin.inst_name}/{qpin_name}"
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(qpin_fullname, neightbor)
                    qindex += 1
                    pin_mapper[qpin_fullname] = pin
                else:
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(new_pin.full_name, neightbor)
                    pin_mapper[new_pin.full_name] = pin
                G.remove_node(pin.full_name)
            del self.ffs[inst.name]
        self.ffs[new_inst.name] = new_inst
        self.new_ffs.append(new_inst)
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

    def get_ffs_names(self):
        return tuple(ff.name for ff in self.get_ffs())

    def get_merged_ffs(self):
        return self.new_ffs

    def get_gates(self):
        return [inst for inst in self.setting.instances if not inst.is_ff]

    def get_library(self):
        return self.setting.library

    def scoring(self):
        total_tns = 0
        total_power = 0
        total_area = 0
        for node, data in self.G.nodes(data="pin"):
            slack = self.timing_slack(node)
            if slack < 0:
                total_tns += -slack
        for ff in self.get_ffs():
            total_power += ff.lib.power
            total_area += ff.lib.area

        return (
            self.setting.alpha * total_tns
            + self.setting.beta * total_power
            + self.setting.gamma * total_area
        )

    def original_pin_distance(self, node1, node2):
        return cityblock(self.get_origin_pin(node1).pos, self.get_origin_pin(node2).pos)

    def current_pin_distance(self, node1, node2):
        return cityblock(self.get_pin(node1).pos, self.get_pin(node2).pos)

    def transfer_graph_to_setting(self, visualized=True, show_distance=False, extension="html"):
        G = self.G
        setting = self.setting
        setting.instances = []
        for name, data in G.nodes(data="pin"):
            if data.is_io:
                continue
            else:
                setting.instances.append(data.inst)
        setting.nets = []
        for node1, node2 in G.edges():
            if self.get_inst(node1) == self.get_inst(node2) and not self.get_inst(node1).is_ff:
                continue
            net = Net("", num_pins=2)
            net.pins = [self.get_pin(node1), self.get_pin(node2)]
            if show_distance:
                net.metadata = cityblock(net.pins[0].pos, net.pins[1].pos)
            setting.nets.append(net)
        if visualized:
            visualize(
                setting,
                file_name=f"output{self.graph_num}.{extension}",
                resolution=None if extension == "html" else 50000,
            )
            self.graph_num += 1

    def print_graph(G):
        for node, data in G.nodes(data="pin"):
            print(node, list(G.neighbors(node)))

    def legalization(self):
        def cityblock_variable(model, v1, v2, bias):
            delta_xy = model.addMVar(2, lb=-GRB.INFINITY)
            abs_delta_xy = model.addMVar(2)
            cityblock_distance = model.addVar(lb=-GRB.INFINITY)
            model.addConstr(delta_xy[0] == v1[0] - v2[0])
            model.addConstr(delta_xy[1] == v1[1] - v2[1])
            model.addConstr(abs_delta_xy[0] == gp.abs_(delta_xy[0]))
            model.addConstr(abs_delta_xy[1] == gp.abs_(delta_xy[1]))
            model.addConstr(cityblock_distance == gp.quicksum(abs_delta_xy) + bias)
            return cityblock_distance

        model = gp.Model("")
        model.setParam("LogToConsole", 0)
        ff_vars = {}
        for ff in self.get_ffs():
            ff_vars[ff.name] = model.addMVar(2, name=ff.name)
        min_negative_slack_vars = []
        for ff in self.get_ffs():
            negative_slack_vars = []
            for curpin in ff.dpins:
                ori_slack = self.get_origin_pin(curpin).slack
                prev_pin = self.get_prev_pin(curpin)
                if prev_pin:
                    current_pin = self.get_pin(curpin)
                    current_pin_pos = [
                        a + b for a, b in zip(ff_vars[current_pin.inst.name], current_pin.rel_pos)
                    ]
                    dpin_pin = self.get_pin(prev_pin)
                    dpin_pin_pos = dpin_pin.pos
                    ori_distance = self.original_pin_distance(prev_pin, curpin)
                    prev_pin_displacement_delay = cityblock_variable(
                        model, current_pin_pos, dpin_pin_pos, -ori_distance
                    )
                else:
                    prev_pin_displacement_delay = 0
                displacement_distances = []
                for pff, qpin in self.get_prev_ffs(curpin):
                    pff_pin = self.get_pin(pff)
                    qpin_pin = self.get_pin(qpin)
                    pff_pos = [a + b for a, b in zip(ff_vars[pff_pin.inst.name], pff_pin.rel_pos)]
                    if qpin_pin.is_ff:
                        qpin_pos = [
                            a + b for a, b in zip(ff_vars[qpin_pin.inst.name], qpin_pin.rel_pos)
                        ]
                    else:
                        qpin_pos = qpin_pin.pos
                    ori_distance = self.original_pin_distance(pff, qpin)
                    distance_var = cityblock_variable(model, pff_pos, qpin_pos, -ori_distance)
                    displacement_distances.append(distance_var)
                if len(displacement_distances) > 0:
                    min_displacement_distance = model.addVar(lb=-GRB.INFINITY)
                    model.addConstr(min_displacement_distance == gp.min_(displacement_distances))
                else:
                    min_displacement_distance = 0

                slack_var = model.addVar(lb=-GRB.INFINITY)
                model.addConstr(
                    slack_var
                    == ori_slack - (prev_pin_displacement_delay + min_displacement_distance)
                )
                negative_slack_var = model.addVar(lb=-GRB.INFINITY)
                model.addConstr(negative_slack_var == gp.min_(0, slack_var))
                negative_slack_vars.append(negative_slack_var)

            if len(negative_slack_vars) > 1:
                min_negative_slack_var = model.addVar(
                    name=f"min_negative_slack for {curpin}", lb=-GRB.INFINITY
                )
                model.addConstr(min_negative_slack_var == gp.min_(negative_slack_vars))
                min_negative_slack_vars.append(min_negative_slack_var)
            else:
                min_negative_slack_vars.append(negative_slack_vars[0])

        model.setObjective(-gp.quicksum(min_negative_slack_vars), gp.GRB.MINIMIZE)
        model.optimize()
        for name, ff_var in ff_vars.items():
            self.get_ffs(name)[0].moveto((ff_var.X[0], ff_var.X[1]))

def get_pin_name(node_name):
    return node_name.split("/")[1]
