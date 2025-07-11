import re
from collections import defaultdict

input_file = "results/workflow_search/aime24/single_baseline_multiple_times_attempt_2/cot/gpt-4.1-mini_o4-mini_oracle.results"
output_file = "merged_output.txt"

# Dictionary để lưu max acc_oracle_verifier_list theo experiment
exp_to_max_acc = defaultdict(int)
# Đọc file
with open(input_file, "r") as f:
    for line in f:
        match = re.search(r"experiemnt (\d+):.*?acc_oracle_verifier_list: \[(\d+)\]", line)
        print(line)
        if match:
            exp_id = int(match.group(1))
            acc_val = int(match.group(2))
            # if exp_id < 185:
            #     continue
            print(exp_id, acc_val)
            exp_to_max_acc[exp_id] = max(exp_to_max_acc[exp_id], acc_val)
            

# Ghi kết quả ra file
with open(output_file, "w") as f:
    for exp_id in sorted(exp_to_max_acc):
        f.write(f"experiemnt {exp_id}: acc_oracle_verifier_list: [{exp_to_max_acc[exp_id]}]\n")
