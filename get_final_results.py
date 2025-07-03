import os
import json

directory = "results/workflow_search/gpqa_diamond/abstract_workflow_refined/gpt-4.1-mini_o4-mini"
total_score = 0
total_time = 0
max_cost = float('-inf')

cnt = 0

file_name = [0] * 198

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        file_name.append(filename)
        number = int(filename.split('_')[-1].split('.')[0])
        file_name[number] = 1
        
        if number < 150:
            continue
        
        cnt += 1
        # print(filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            max_score = 0
            for dat in data:
                max_score = max(dat.get("score", 0), max_score)
                total_time += dat.get("total_time", 0)
                max_cost = max(max_cost, dat.get("max_cost", float('-inf')))
                
            total_score += max_score
            
print(cnt)
for idx in range(0, 198):
    if file_name[idx] == 0:
        print(idx)

print(f"Total score: {total_score / cnt * 100}")
print(f"Average time: {total_time / cnt}")
print(f"Average cost: {max_cost / cnt}")
