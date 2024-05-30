import copy
import itertools
from pprint import pprint

import gurobipy as gp
import networkx as nx
import numpy as np
from scipy.spatial.distance import cityblock

from input import Inst, Net, PhysicalPin, read_file, visualize

print_tmp = print
print = lambda *args: (
    print_tmp(*args) if len(args) > 1 else pprint(args[0]) if args else print_tmp()
)
input_path = "cases/sample.txt"
input_path = "cases/sampleCase"
# input_path = "v1.txt"
input_path = "v2.txt"
setting = read_file(input_path)


# print(setting.alpha, setting.beta, setting.gamma, setting.lambde)
# pprint(setting.instances)
# pprint(setting.flip_flops)


class MBFFG:
    def __init__(self, setting):
        G = nx.Graph()
        pin_mapper = {}
        for inst in setting.instances:
            if not inst.is_ff:
                G.add_edges_from(itertools.combinations([pin.full_name for pin in inst.pins], 2))
            for pin in inst.pins:
                G.add_node(pin.full_name, pin=pin)
                pin_mapper[pin.full_name] = copy.deepcopy(pin)
        for input in setting.inputs:
            for pin in input.pins:
                G.add_node(pin.full_name, pin=pin)
                pin_mapper[pin.full_name] = input
        for output in setting.outputs:
            for pin in output.pins:
                G.add_node(pin.full_name, pin=pin)
                pin_mapper[pin.full_name] = output
        for net in setting.nets:
            output_pin = net.pins[0]
            for pin in net.pins[1:]:
                G.add_edge(output_pin.full_name, pin.full_name)
        self.G = G
        self.pin_mapper = pin_mapper
        self.setting = setting
        self.graph_num = 1
        for inst in setting.instances:
            if inst.is_ff:
                for pin in inst.pins:
                    if pin.name.startswith("d") and pin.slack is None:
                        ff0s = self.get_prev_ffs(pin.full_name)
                        print(self.get_prev_ffs_path(pin.full_name))
                        exit()
                        path_slacks = np.zeros(len(ff0s))
                        for i, (ff0, c0) in enumerate(ff0s):
                            path_slacks[i] = (
                                cityblock(self.get_pin(ff0).pos, self.get_pin(c0).pos)
                                * setting.displacement_delay
                            ) + self.get_inst(ff0).qpin_delay
                        c1 = self.get_prev_pin(pin.full_name)
                        if c1:
                            path_slacks += cityblock(
                                self.get_pin(c1).pos, self.get_pin(pin.full_name).pos
                            )
                        print(path_slacks)
                        print(c1)

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

    def get_prev_ffs(self, node_name):
        prev_ffs = set()
        waiting_queue = [node_name]
        pioneers = set()
        while waiting_queue:
            node = waiting_queue.pop(0)
            pioneers.add(node)
            for neighbor in self.G.neighbors(node):
                neighbor_pin = self.get_pin(neighbor)
                if neighbor in pioneers:
                    continue
                if neighbor_pin.is_io:
                    continue
                if neighbor_pin.is_ff:
                    prev_ffs.add((neighbor, node))
                else:
                    waiting_queue.append(neighbor)
        return list(prev_ffs)

    def get_prev_pin(self, node_name):
        prev_pins = []
        for neighbor in self.G.neighbors(node_name):
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io or (not neighbor_pin.is_ff):
                prev_pins.append(neighbor)
        assert len(prev_pins) <= 1, f"Multiple previous pins for {node_name}"
        if not prev_pins:
            return None
        else:
            return prev_pins[0]

    def get_prev_ffs_path(self, node_name, temp_path=[], full_path=[]):
        for neighbor in self.G.neighbors(node_name):
            neighbor_pin = self.get_pin(neighbor)
            if neighbor_pin.is_io:
                continue
            if neighbor_pin.is_ff:
                full_path.append(temp_path + [neighbor])
            else:
                return self.get_prev_ffs_path(neighbor, temp_path + [neighbor], full_path)
        return full_path

    def timing_slack(self, node_name):
        if self.get_pin(node_name).slack is None:
            print("No slack for", node_name)
            return 0
        prev_pin = self.get_prev_pin(node_name)
        dpin_node_pos = self.get_pin(prev_pin).pos
        self_displacement_delay = (
            (
                cityblock(
                    dpin_node_pos,
                    self.get_origin_pin(node_name).pos,
                )
                - cityblock(dpin_node_pos, self.get_pin(node_name).pos)
            )
            if prev_pin
            else 0
        )
        prev_ffs = self.get_prev_ffs(node_name)
        prev_ffs_qpin_displacement_delay = [float("inf")]
        for pff, qpin in prev_ffs:
            prev_ffs_qpin_displacement_delay.append(
                self.get_origin_inst(pff).qpin_delay
                - self.get_inst(pff).qpin_delay
                + cityblock(
                    self.get_origin_pin(pff).pos,
                    self.get_origin_pin(qpin).pos,
                )
                - cityblock(self.get_pin(pff).pos, self.get_pin(qpin).pos)
            )
        # print(prev_ffs_qpin_displacement_delay)
        total_delay = (
            self.get_origin_pin(node_name).slack
            + self_displacement_delay
            + min(prev_ffs_qpin_displacement_delay)
        )
        # print(prev_ffs)
        # print(prev_pins)
        return total_delay

    def merge_ff(self, insts, lib):
        if isinstance(insts, str):
            insts = self.get_insts(insts.split(","))
        G = self.G
        pin_mapper = self.pin_mapper
        assert sum([inst.lib.bits for inst in insts]) == setting.library[lib].bits
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

        return new_inst

    def transfer_graph_to_setting(self, visualized=True, show_distance=False):
        G = self.G
        setting = self.setting
        setting.instances = []
        for _, data in G.nodes(data="pin"):
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
            visualize(setting, file_name=f"output{self.graph_num}.html")
            self.graph_num += 1

    def print_graph(G):
        for node, data in G.nodes(data="pin"):
            print(node, list(G.neighbors(node)))


def get_pin_name(node_name):
    return node_name.split("/")[1]


# print(G.edges("reg4/D"))
# print(G.edges("reg4/Q"))
# print(timing_slack(G, "reg3/d"))
# print(G.nodes["c2/d"])
# G.nodes["c2/d"]["pos"] = np.array(G.nodes["c2/d"]["pos"]) + np.array([0, 1])
# print(G.nodes["c2/d"]["pos"])
# print(G.nodes["c2/d"]["pin"].pos)
# G.nodes["c1/q"]["pin"].inst.x += 1
# G.nodes["c1/q"]["pin"].inst.y += 1
# print(G.nodes["c2/d"]["pin"].pos)
# print(timing_slack(G, "c2/d"))

# G.nodes["c1/d"]["pin"].inst.x += 1
# print(timing_slack(G, "c5/in"))
# print(timing_slack(G, "c2/d"))

# inst = merge_ff(G, get_insts(G, "c1/d,c2/d,c3/d,c4/d".split(",")), "ff4")
# inst.x = 20
mbffg = MBFFG(setting)
# mbffg.transfer_graph_to_setting()
# mbffg.merge_ff("c1/q", "ff1e")
# print(mbffg.get_prev_ffs("c2/d"))
mbffg.get_inst("c1/d").x -= 1
print(mbffg.timing_slack("c2/d"))
mbffg.transfer_graph_to_setting()
# oinst = get_inst(G, ["c1/d"])[0]
# ninst = merge_ff(G, [oinst], "ff1e")
# ninst.x = oinst.x
# ninst.y = oinst.y
# print(timing_slack(G, "c2/d"))
# transfer_graph_to_setting(G, setting)
