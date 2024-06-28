from mbffg import MBFFG, VisualizeOptions
from utility import *

input_path = "../cases/new_c1.txt"
options = VisualizeOptions(
    line=False,
    cell_text=False,
    io_text=False,
    placement_row=False,
)
mbffg = MBFFG(input_path)
mbffg.transfer_graph_to_setting(options=options, visualized=False)
mbffg.merge_ff("FLIP_FLOP27,FLIP_FLOP28", "FF2")
mbffg.merge_ff("FLIP_FLOP32,FLIP_FLOP33", "FF2")
mbffg.merge_ff("FLIP_FLOP15,FLIP_FLOP24", "FF2")
mbffg.merge_ff("FLIP_FLOP62,FLIP_FLOP72", "FF2")
mbffg.legalization()
final_score = mbffg.scoring()
print(f"score: {final_score}")
mbffg.transfer_graph_to_setting(options=options, visualized=False)
# 結果輸出成輸入的格式
mbffg.log("temp.out")

# 輸入temp.out
mbffg = MBFFG("temp.out")
mbffg.merge_ff("FF2_FLIP_FLOP15_FLIP_FLOP24,FF2_FLIP_FLOP62_FLIP_FLOP72", "FF4")
mbffg.legalization()
final_score = mbffg.scoring()
mbffg.transfer_graph_to_setting(options=options, visualized=False)
# mbffg.log("temp.out")
print(f"score: {final_score}")
