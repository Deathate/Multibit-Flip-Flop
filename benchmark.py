from mbffg import MBFFG, VisualizeOptions
from faketime_utl import ensure_time
ensure_time()
files = ["c1", "c2", "c3", "c4", "c5"][:1]
scores = []
for file in files:
    input_path = f"cases/new_{file}.txt"
    mbffg = MBFFG(input_path)
    with open(f"output/{file}.out") as f:
        # [FLIP_FLOP_LIST]
        # FF4 ff_NEW_FLIP_FLOP_4,NEW_FLIP_FLOP_2 (460,160)
        # FF4 ff_NEW_FLIP_FLOP_6,NEW_FLIP_FLOP_5 (350,250)
        # FF4 ff_NEW_FLIP_FLOP_1,NEW_FLIP_FLOP_7 (535,450)
        # FF4 ff_NEW_FLIP_FLOP_8,NEW_FLIP_FLOP_3 (760,515)
        for line in f.readlines():
            line = line.strip()
            if line.startswith("FF"):
                library, names = line.split(" ")[:2]
                # names = names.split(",")
                names = names[3:]
                mbffg.merge_ff(names, library)
    mbffg.legalization()
    scores.append(mbffg.scoring())
print(scores)
# [93719.23764227911, 368819.1374475311, 1472254.3339172646, 4513756.821312344, 9190230.42190052]
