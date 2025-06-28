import os
import json

directory = "results/workflow_search/aime24/abstract_workflow_v9/gpt-4.1-mini_gpt-4o-mini-2024-07-18"
total_score = 0
total_time = 0
max_cost = float('-inf')

cnt = 0

file_name = [0] * 30

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        cnt += 1
        # print(filename)
        file_name.append(filename)
        number = int(filename.split('_')[-1].split('.')[0])
        file_name[number] = 1
        with open(filepath, "r") as f:
            data = json.load(f)
            max_score = 0
            for dat in data:
                max_score = max(dat.get("score", 0), max_score)
                total_time += dat.get("total_time", 0)
                max_cost = max(max_cost, dat.get("max_cost", float('-inf')))
                
            total_score += max_score
            
print(cnt)
for idx in range(0, 30):
    if file_name[idx] == 0:
        print(idx)

print(f"Total score: {total_score / cnt * 100}")
print(f"Total time: {total_time / cnt}")
print(f"Max cost: {max_cost / cnt}")
