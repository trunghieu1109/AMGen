import os
import json
import re

# Thư mục chứa các file JSON gốc
input_dir = 'workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v3_test/mas'  # Thay bằng đường dẫn thực tế

# Regex để bắt {x} từ tên file
pattern = re.compile(r"mas_zero_workflow_analysis_(\d+)_iteration_0\.json")

# Duyệt qua các file trong thư mục
for filename in os.listdir(input_dir):
    match = pattern.match(filename)
    if match:
        x = match.group(1)
        input_path = os.path.join(input_dir, filename)
        output_filename = f"mas_{x}.json"
        output_path = os.path.join(input_dir, output_filename)
        
        # Đọc dữ liệu từ file gốc
        with open(input_path, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
        
        # Ghi ra file mới với tên đã đổi
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2, ensure_ascii=False)

        print(f"Đã chuyển: {filename} -> {output_filename}")
