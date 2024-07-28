import re
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace

name = "c5"
input_path = f"cases/{name}.in"
output_path = f"cases/new_{name}.txt"
Alpha = 1
Beta = 5
Gamma = 5
Lambda = 1
DisplacementDelay = 0.01
DieSize = [0, 0, 0, 0]
NumInput = 0
Input = []
NumOutput = 0
Output = []
BinWidth = 0
BinHeight = 0
BinMaxUtil = 79
GridSIZE = [0, 0]
library = []
inst = []
input = []
output = []
netlist = []
block = []
# CHIP_SIZE 6000 x 6000
# GRID_SIZE 5 x 5
# BIN_SIZE 500 x 500
# [FLIP_FLOP FF1]
# BIT_NUMBER 1
# POWER_CONSUMPTION 100
# AREA 100
# FF1 FLIP_FLOP3_#1_3 (490,1650)
# FF1 FLIP_FLOP4_#1_4 (245,1270)
# FF1 FLIP_FLOP6_#1_6 (1675,2415)
# FF1 FLIP_FLOP8_#1_8 (1635,1435)
# FF1 FLIP_FLOP9_#1_9 (590,2490)
# INPUT PIN1_#1_1 (515,1785)
# OUTPUT PIN2_#1_2 (485,2085)
# PIN1_# 1_1 NEW_FLIP_FLOP_22 490
# PIN2_# 1_2 NEW_FLIP_FLOP_22 900
lib_query = {}
inst_query = {}
input_set = set()
output_set = set()
with open(input_path, "r") as file:
    for line in file.readlines():
        line = line.strip()
        if line.startswith("CHIP_SIZE"):
            DieSize[2], DieSize[3] = map(int, line[line.index(" ") :].split("x"))
        elif line.startswith("GRID_SIZE"):
            GridSIZE[0], GridSIZE[1] = map(int, line[line.index(" ") :].split("x"))
        elif line.startswith("BIN_SIZE"):
            BinWidth, BinHeight = map(int, line[line.index(" ") :].split("x"))
        elif line.startswith("[FLIP_FLOP "):
            library.append(SimpleNamespace(name=line[line.index(" ") + 1 : -1]))
            lib_query[library[-1].name] = library[-1]
        elif line.startswith("BIT_NUMBER"):
            library[-1].bit = int(line[line.index(" ") :])
        elif line.startswith("POWER_CONSUMPTION"):
            library[-1].power = int(line[line.index(" ") :])
        elif line.startswith("AREA"):
            library[-1].area = int(line[line.index(" ") :])
        elif line.startswith("FF"):
            inst.append(
                SimpleNamespace(
                    lib=line[: line.index(" ")],
                    name=line[line.index(" ") + 1 : line.index(" (")],
                    pos=(
                        int(line[line.index("(") + 1 : line.index(",")]),
                        int(line[line.index(",") + 1 : -1]),
                    ),
                )
            )
            inst_query[inst[-1].name] = inst[-1]
        elif line.startswith("INPUT"):
            input.append(
                SimpleNamespace(
                    name=line[line.index(" ") + 1 : line.index(" (")],
                    pos=(
                        int(line[line.index("(") + 1 : line.index(",")]),
                        int(line[line.index(",") + 1 : -1]),
                    ),
                )
            )
            input_set.add(input[-1].name)
        elif line.startswith("OUTPUT"):
            output.append(
                SimpleNamespace(
                    name=line[line.index(" ") + 1 : line.index(" (")],
                    pos=(
                        int(line[line.index("(") + 1 : line.index(",")]),
                        int(line[line.index(",") + 1 : -1]),
                    ),
                )
            )
            output_set.add(output[-1].name)
        elif line.startswith("PIN"):
            netlist.append(line.split(" ")[:2])
        elif line.startswith("BLOCK"):
            block.append(
                SimpleNamespace(
                    name=line[: line.index(" ")],
                    pos=(
                        int(line[line.index("(") + 1 : line.index(",")]),
                        int(line[line.index(",") + 1 : line.index(")")]),
                    ),
                    area=int(line[line.index(")") + 1 :]),
                ),
            )
# net_all = [net[0] for net in netlist] + [net[1] for net in netlist]
net_ctr_d = defaultdict(int)
net_ctr_q = defaultdict(int)
with open(output_path, "w") as file:
    file.write(f"Alpha {Alpha}\n")
    file.write(f"Beta {Beta}\n")
    file.write(f"Gamma {Gamma}\n")
    file.write(f"Lambda {Lambda}\n")
    file.write(f"DieSize {DieSize[0]} {DieSize[1]} {DieSize[2]} {DieSize[3]}\n")
    file.write(f"NumInput {len(input)}\n")
    for i in input:
        file.write(f"Input {i.name} {i.pos[0]} {i.pos[1]}\n")
    file.write(f"NumOutput {len(output)}\n")
    for i in output:
        file.write(f"Output {i.name} {i.pos[0]} {i.pos[1]}\n")
    for lib in library:
        width = lib.area**0.5
        file.write(f"FlipFlop {lib.bit} {lib.name} {width} {width} {lib.bit*2}\n")
        for i in range(lib.bit):
            file.write(f"Pin D{i} 0 {width/2}\n")
            file.write(f"Pin Q{i} {width} {width/2}\n")
        # file.write(f"Pin f0 0 0\n")
        # file.write(f"Pin f1 0 0\n")
        # file.write(f"Pin f2 0 0\n")
    for b in block:
        width = b.area**0.5
        file.write(f"Gate {b.name} {width} {width} 2\n")
        file.write(f"Pin IN 0 {width/2}\n")
        file.write(f"Pin OUT {width} {width/2}\n")
    file.write(f"NumInstances {len(block)+len(inst)}\n")
    for b in block:
        file.write(f"Inst {b.name} {b.name} {b.pos[0]} {b.pos[1]}\n")
    for i in inst:
        file.write(f"Inst {i.name} {i.lib} {i.pos[0]} {i.pos[1]}\n")
    file.write(f"NumNets {len(netlist)}\n")
    for i, n in enumerate(netlist):
        file.write(f"Net n{i} 2\n")
        file.write(f"Pin {n[0]}\n")
        # if net_ctr[n[1]] >= (bit:=lib_query[inst_query[n[1]].lib].bit):
        #     file.write(f"Pin {n[1]}/f{net_ctr[n[1]]-bit}\n")
        # else:
        if n[0] in input_set:
            file.write(f"Pin {n[1]}/D{net_ctr_d[n[1]]}\n")
            net_ctr_d[n[1]] += 1
        elif n[0] in output_set:
            file.write(f"Pin {n[1]}/Q{net_ctr_q[n[1]]}\n")
            net_ctr_q[n[1]] += 1
        else:
            raise ValueError(f"Net {n[0]} not found")
    file.write(f"BinWidth {BinWidth}\n")
    file.write(f"BinHeight {BinHeight}\n")
    file.write(f"BinMaxUtil {BinMaxUtil}\n")
    for i in range(0, DieSize[3], 10):
        file.write(f"PlacementRows 0 {i} 2 10 {int(DieSize[2]/2)}\n")
    file.write(f"DisplacementDelay {DisplacementDelay}\n")
    # QpinDelay FF1 1.0
    for lib in library:
        file.write(f"QpinDelay {lib.name} 1.0\n")
    for b in inst:
        number = re.findall(r"\d+", b.lib)
        number = int(number[0])
        for i in range(number):
            file.write(f"TimingSlack {b.name} D{i} 0\n")
    # GatePower FF1e 10.0
    for lib in library:
        file.write(f"GatePower {lib.name} {lib.power}\n")

    # file.write(f"GridSIZE = {GridSIZE}\n")
    # file.write(f"inst = {inst}\n")
    # file.write(f"input = {input}\n")
    # file.write(f"output = {output}\n")
    # file.write(f"netlist = {netlist}\n")
