from pprint import pprint

from mbffg import MBFFG

input_path = "cases/new_c1.txt"
mbffg = MBFFG(input_path)
ori_score = mbffg.scoring()
# get the flip-flops and library
pprint(mbffg.get_ffs())
pprint(mbffg.get_library())
print(mbffg.get_prev_pin("new_flip_flop_1/d0"))
print(mbffg.get_pin("pin41").pos)
# merge the flip-flops with their names and the target library
mbffg.merge_ff("new_flip_flop_1,new_flip_flop_2", "ff4")
# legalize the design before scoring
mbffg.legalization()
final_score = mbffg.scoring()
print(ori_score, final_score)
# transfer the graph to a setting file and save it as svg
mbffg.transfer_graph_to_setting(extension="svg")
