import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.join(CUR_DIR, "../"))


dict = {}
with open(os.path.join(PARENT_DIR, "timing7.log"), "r") as f:
    lines = f.readlines()
    split_lines = [line.split() for line in lines]
    if len(split_lines) > 0:
        for line in split_lines:
            if len(line) >= 5:
                token = int(line[0].split(":")[1].strip(","))
                layer = int(line[1].split(":")[1].strip("]"))
                message = line[2]
                time = float(line[3])
                if message not in dict:
                    dict[message] = []
                dict[message].append((token, layer, time))

for i in dict:
    times = dict[i]
    total_time = sum([t[2] for t in times])
    avg_time = total_time / len(times)
    print(f"{i}: {avg_time:.9f} seconds (total: {total_time:.9f} seconds, count: {len(times)})")
    print()