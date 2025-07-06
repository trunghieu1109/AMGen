from collections import defaultdict
import re

# Danh sách flow đầu vào (filename, flow)
flows = [
    ("mas_0.json", ['start_sequential', 'start_loop', 1, 6, 'end_loop', 2, 5, 'end_sequential']),
    ("mas_1.json", ['start_sequential', 'start_loop', 1, 6, 'end_loop', 2, 'end_sequential']),
    ("mas_2.json", ['start_sequential', 1, 0, 'end_sequential']),
    ("mas_3.json", ['start_sequential', 'start_loop', 1, 'end_loop', 2, 0, 'end_sequential']),
    ("mas_4.json", ['start_sequential', 'start_loop', 1, 0, 'end_loop', 2, 'end_sequential']),
    ("mas_5.json", ['start_sequential', 'start_loop', 1, 'end_loop', 2, 0, 5, 'end_sequential']),
    ("mas_8.json", ['start_sequential', 'start_loop', 1, 'end_loop', 2, 0, 3, 'end_sequential']),
    ("mas_9.json", ['start_sequential', 'start_loop', 1, 'end_loop', 2, 0, 4, 'end_sequential']),
    ("mas_10.json", ['start_sequential', 'start_loop', 1, 0, 'end_loop', 2, 5, 'end_sequential']),
]

def patternize(flow):
    """Chuyển flow thành dạng pattern với placeholder cho số nguyên"""
    pattern = []
    var_positions = []
    for idx, item in enumerate(flow):
        if isinstance(item, int):
            pattern.append("{X}")
            var_positions.append(idx)
        else:
            pattern.append(item)
    return tuple(pattern), tuple(var_positions)

# Gom nhóm theo pattern
grouped = defaultdict(lambda: {"variants": [], "files": []})

for filename, flow in flows:
    pattern, var_pos = patternize(flow)
    vars_only = tuple(flow[i] for i in var_pos)
    key = (pattern, var_pos)
    grouped[key]["variants"].append(vars_only)
    grouped[key]["files"].append(filename)

# In kết quả nhóm
for i, ((pattern, var_pos), data) in enumerate(grouped.items()):
    print(f"\nGroup {i+1}:")
    print("  Pattern:", pattern)
    print("  Variable positions:", var_pos)
    print("  Variants:", data["variants"])
    print("  Files:", data["files"])
