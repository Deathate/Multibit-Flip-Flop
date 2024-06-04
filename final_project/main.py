
from mbffg import MBFFG

input_path = "new_c1.txt"

mbffg = MBFFG(input_path)
ori_score = mbffg.scoring()
# 合併不同的flip-flop會得到不同的分數
mbffg.merge_ff("new_flip_flop_9,new_flip_flop_19", "ff4")
# mbffg.merge_ff("flip_flop3,flip_flop4", "ff2")
# mbffg.merge_ff("flip_flop8,flip_flop9", "ff2")
mbffg.legalization()
final_score = mbffg.scoring()
print(ori_score - final_score)
mbffg.transfer_graph_to_setting(extension="svg")
