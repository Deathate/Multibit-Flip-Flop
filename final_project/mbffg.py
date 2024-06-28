import copy
import itertools
import time
from collections import defaultdict
from functools import cache, cached_property
from pprint import pprint

import gurobipy as gp
import networkx as nx
import numpy as np
import rtree
import shapely
import shapely.ops as ops
from gurobipy import GRB
from scipy.spatial.distance import cityblock
from shapely.geometry import Point
from tqdm.auto import tqdm

from input import Inst, Net, PhysicalPin, PlotlyUtility, VisualizeOptions, read_file, visualize

# import graphx as nx


print_tmp = print


def print(*args):
    print_tmp(*args) if len(args) > 1 else pprint(args[0]) if args else print_tmp()


class MBFFG:
    def __init__(self, file_path):
        print("Reading file...")
        setting = read_file(file_path)
        print("File read")
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
                pin.slack = 0
        for net in setting.nets:
            output_pin = net.pins[0]
            for pin in net.pins[1:]:
                G.add_edge(output_pin.full_name, pin.full_name)
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
        self.new_ffs = []
        for inst in self.setting.instances:
            if inst.is_ff:
                assert [self.get_pin(dpin).slack for dpin in inst.dpins].count(
                    None
                ) == 0, f"FF {inst.name} has None slack"
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
    def c_get_prev_ffs_path(self, node_name):
        node_pin = self.get_pin(node_name)
        if node_pin.is_gt:
            if node_pin.is_out:
                neightbors = [n for n in self.get_inst(node_name).inpins]
            elif node_pin.is_in:
                # neightbors = []
                # for n in self.G.neighbors(node_name):
                #     if self.get_inst(n).name != node_pin.inst.name:
                #         neightbors.append(n)
                neightbors = [
                    n
                    for n in self.G.neighbors(node_name)
                    if self.get_inst(n).name != node_pin.inst.name
                ]
            else:
                raise ValueError(f"Unknown gate type {node_pin}")
        else:
            neightbors = list(self.G.neighbors(node_name))

        result = set()
        for neighbor in neightbors:
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io:
                pass
            elif neighbor_pin.is_ff:
                if neighbor_pin.is_q:
                    result.add((neighbor, node_name))
                    # self.c_get_prev_ffs_path_result_cache.append((neighbor, node_name))
            else:
                result.update(self.c_get_prev_ffs_path(neighbor))
        return set(result)

    def get_prev_ffs(self, node_name):
        return self.c_get_prev_ffs_path(node_name)
        # return [(n[0], n[1]) for n in self.get_prev_ffs_path(node_name)]

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
            # neighbor_pin = self.get_pin(neighbor)
            res.append(neighbor)
        return res

    # def get_ff_neighbor_inst_pin(self, node_name):
    #     inst = self.get_ffs(node_name)[0]
    #     res = []
    #     for dpin in inst.dpins:
    #         res.extend(self.get_prev_inst_pin(dpin))
    #     for qpin in inst.qpins:
    #         res.extend(self.get_fol_inst_pins(qpin))
    #     return res

    # def min_distance_to_neightbor_inst_pin(self, node_name):
    #     inst = self.get_ffs(node_name)[0]
    #     min_distance = float("inf")
    #     min_before_distance = float("inf")
    #     pin = ""
    #     for dpin in inst.dpins:
    #         for pin in self.get_prev_inst_pin(dpin):
    #             min_distance = min(
    #                 min_distance, cityblock(self.get_pin(dpin).pos, self.get_pin(pin).pos)
    #             )
    #             if min_distance < min_before_distance:
    #                 min_before_distance = min_distance
    #                 pin = pin
    #     for qpin in inst.qpins:
    #         for pin in self.get_fol_inst_pins(qpin):
    #             min_distance = min(
    #                 min_distance, cityblock(self.get_pin(qpin).pos, self.get_pin(pin).pos)
    #             )
    #             if min_distance < min_before_distance:
    #                 min_before_distance = min_distance
    #                 pin = pin
    #     return min_distance, pin

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
        # print(
        #     node_name,
        #     total_delay,
        #     self.get_origin_pin(node_name).slack,
        #     self_displacement_delay,
        #     min(prev_ffs_qpin_displacement_delay),
        # )
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
                if pin.is_d:
                    dpin_fullname = new_pin.inst.dpins[dindex]
                    for neightbor in G.neighbors(pin.full_name):
                        G.add_edge(dpin_fullname, neightbor)
                    dindex += 1
                    # new_inst.pins_query[dpin_name].slack = pin.slack
                    pin_mapper[dpin_fullname] = pin
                elif pin.is_q:
                    # qpin_fullname = f"{new_pin.inst_name}/{qpin_name}"
                    qpin_fullname = new_pin.inst.qpins[qindex]
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

    @cache
    def maximum_bits_of_library(self):
        return max([lib.bits for lib in self.setting.library.values()])

    def scoring(self):
        print("Scoring...")
        total_tns = 0
        total_power = 0
        total_area = 0
        for ff in tqdm(self.get_ffs()):
            slacks = [self.timing_slack(dpin) for dpin in ff.dpins]
            min_slack = min(min(slacks), 0)
            # if min_slack < 0:
            #     print(f"Negative slack {min_slack} for {ff.name}")
            total_tns += -min_slack
            total_power += ff.lib.power
            total_area += ff.lib.area
        print("Scoring done")
        return (
            self.setting.alpha * total_tns
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
            if self.get_inst(node1) == self.get_inst(node2) and not self.get_inst(node1).is_ff:
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

    def legalization(self):
        self.c_get_prev_ffs_path.cache_clear()

        def cityblock_variable(model, v1, v2, bias, weight, intercept):
            delta_xy = model.addMVar(2, lb=-GRB.INFINITY)
            abs_delta_xy = model.addMVar(2)
            cityblock_distance = model.addVar(lb=-GRB.INFINITY)
            model.addConstr(delta_xy[0] == v1[0] - v2[0])
            model.addConstr(delta_xy[1] == v1[1] - v2[1])
            model.addConstr(abs_delta_xy[0] == gp.abs_(delta_xy[0]))
            model.addConstr(abs_delta_xy[1] == gp.abs_(delta_xy[1]))
            model.addConstr(
                cityblock_distance == weight * (gp.quicksum(abs_delta_xy) + bias) + intercept
            )
            return cityblock_distance

        print("Legalizing...")
        with gp.Env(empty=True) as env:
            env.setParam("LogToConsole", 0)
            env.start()
            with gp.Model(env=env) as model:
                ff_vars = {}
                for ff in self.get_ffs():
                    ff_vars[ff.name] = model.addMVar(2, name=ff.name)
                min_negative_slack_vars = []
                dis2ori_locations = []
                for ff in tqdm(self.get_ffs()):
                    negative_slack_vars = []
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
                        for pff, qpin in self.get_prev_ffs(curpin):
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
                        negative_slack_vars.append(slack_var)

                    if len(negative_slack_vars) > 1:
                        min_negative_slack_var = model.addVar(
                            name=f"min_negative_slack for {ff.name}", lb=-GRB.INFINITY
                        )
                        model.addConstr(min_negative_slack_var == gp.min_(negative_slack_vars, 0))
                        min_negative_slack_vars.append(min_negative_slack_var)
                    else:
                        min_negative_slack_vars.append(negative_slack_vars[0])
                    dis2ori = cityblock_variable(model, ff_vars[ff.name], ff.pos, 0, 1, 0)
                    dis2ori_locations.append(dis2ori)
                model.setObjectiveN(-gp.quicksum(min_negative_slack_vars), 0, priority=1)
                model.setObjectiveN(gp.quicksum(dis2ori_locations), 1, priority=0)
                model.optimize()
                # print(model.getObjective().getValue())

                for name, ff_var in ff_vars.items():
                    self.get_ffs(name)[0].moveto((ff_var.X[0], ff_var.X[1]))
                for name in ff_vars:
                    ff_vars[name] = self.get_ffs(name)[0].pos
            print("Legalized")
            # a = time.time()

            # def generator_function(somedata):
            #     for i, obj in enumerate(somedata):
            #         rect = (obj[0], obj[1], obj[0] + obj[2], obj[1] + obj[3])
            #         yield (
            #             i,
            #             rect,
            #             (obj[0], obj[1]),
            #         )

            # points = []
            # for placement_row in self.setting.placement_rows:
            #     for i in range(placement_row.num_cols):
            #         x, y = placement_row.x + i * placement_row.width, placement_row.y
            #         points.append((x, y, placement_row.width, placement_row.height))

            # idx = rtree.index.Index(generator_function(points))
            # for gate in self.get_gates():
            #     for its in idx.intersection(gate.bbox, objects=True):
            #         idx.delete(its.id, its.bbox)

            # for name, ff_var in ff_vars.items():
            #     available_pos = list(
            #         idx.nearest(self.get_ffs(name)[0].bbox, num_results=1, objects=True)
            #     )[0]
            #     self.get_ffs(name)[0].moveto(available_pos.object)
            #     idx.delete(available_pos.id, available_pos.bbox)
            # print(time.time() - a)

        from scipy.spatial import KDTree

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
        while remaining_ffs:
            points = points[~points.mask[:, 0]]
            tree = KDTree(points)
            placed_ffs = []
            for name in remaining_ffs.copy():
                ff_var = ff_vars[name]
                dd, ii = tree.query([ff_var], k=50)
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
                        break
                    else:
                        points[i] = np.ma.masked
                # else:
                #     print("No available position for", name)

            # if remaining_ffs:
            #     remove_points_from_ma(points, tree, [self.get_ffs(name)[0] for name in placed_ffs])

        # print(time.time() - a)

    def log(self, filename):
        setting = self.setting
        with open(filename, "w") as file:
            file.write(f"Alpha {setting.alpha}\n")
            file.write(f"Beta {setting.beta}\n")
            file.write(f"Gamma {setting.gamma}\n")
            file.write(f"Lambda {setting.lambde}\n")
            file.write(
                f"DieSize {setting.die_size.xLowerLeft} {setting.die_size.yLowerLeft} {setting.die_size.xUpperRight} {setting.die_size.yUpperRight}\n"
            )
            file.write(f"NumInput {len(setting.inputs)}\n")
            for i in setting.inputs:
                file.write(f"Input {i.name} {i.x} {i.y}\n")
            file.write(f"NumOutput {len(setting.outputs)}\n")
            for i in setting.outputs:
                file.write(f"Output {i.name} {i.x} {i.y}\n")
            for n in setting.library:
                lib = setting.library[n]
                file.write(
                    f"FlipFlop {lib.bits} {lib.name} {lib.width} {lib.height} {len(lib.pins)}\n"
                )
                for i in lib.pins:
                    file.write(f"Pin {i.name} {i.x} {i.y}\n")
            for b in setting.gates:
                file.write(f"Gate {b.name} {b.width} {b.height} {len(b.pins)}\n")
                for i in b.pins:
                    file.write(f"Pin {i.name} {i.x} {i.y}\n")
            file.write(f"NumInstances {len(setting.instances)}\n")
            for i in setting.instances:
                file.write(f"Inst {i.name} {i.lib.name} {i.x} {i.y}\n")
            file.write(f"NumNets {len(setting.nets)}\n")
            for n in setting.nets:
                file.write(f"Net {n.name} {len(n.pins)}\n")
                for pin in n.pins:
                    file.write(f"Pin {pin.full_name}\n")

            file.write(f"BinWidth {setting.bin_width}\n")
            file.write(f"BinHeight {setting.bin_height}\n")
            file.write(f"BinMaxUtil {setting.bin_max_util}\n")
            for i in setting.placement_rows:
                # file.write(f"PlacementRows 0 {i} 2 10 {int(DieSize[2]/2)}\n")
                file.write(f"PlacementRows {i.x} {i.y} {i.width} {i.height} {i.num_cols}\n")
            file.write(f"DisplacementDelay {setting.displacement_delay}\n")
            # # QpinDelay FF1 1.0
            for n in setting.library:
                lib = setting.library[n]
                file.write(f"QpinDelay {lib.name} {lib.qpin_delay}\n")
            for b in setting.instances:
                for pin in b.pins:
                    if pin.is_d:
                        file.write(
                            f"TimingSlack {b.name} {pin.name} {self.timing_slack(pin.full_name)}\n"
                        )
            # # GatePower FF1e 10.0
            for n in setting.library:
                lib = setting.library[n]
                file.write(f"GatePower {lib.name} {lib.power}\n")


def get_pin_name(node_name):
    return node_name.split("/")[1]
