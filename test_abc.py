import re
from collections import defaultdict

input_file = "results/workflow_search/aime24/dev_31_refactored/gpt-4.1-mini_gpt-4.1-mini_oracle.results"
output_file = "merged_output.txt"

# Dictionary để lưu max acc_oracle_verifier_list theo experiment
exp_to_max_acc = defaultdict(int)
appearance = [0] * 198
# Đọc file
with open(input_file, "r") as f:
    for line in f:
        line = line.replace("0.0", "0")
        line = line.replace("1.0", "1")
        match = re.search(r"experiemnt (\d+):.*?acc_oracle_verifier_list: \[(\d+)\]", line)
        # print(line)
        if match:
            exp_id = int(match.group(1))
            acc_val = int(match.group(2))
            if appearance[exp_id] == 1:
                if acc_val == 1:
                    print(exp_id)
                # continue
            appearance[exp_id] = 1
            # if exp_id < 185:
            #     continue
            # print(exp_id, acc_val)
            exp_to_max_acc[exp_id] = max(exp_to_max_acc[exp_id], acc_val)
            
failed_cases = []
# Ghi kết quả ra file
with open(output_file, "w") as f:
    for exp_id in sorted(exp_to_max_acc):
        if int(exp_to_max_acc[exp_id]) == 0:
            failed_cases.append(exp_id)
        f.write(f"experiemnt {exp_id}: acc_oracle_verifier_list: [{exp_to_max_acc[exp_id]}]\n")

print(failed_cases)