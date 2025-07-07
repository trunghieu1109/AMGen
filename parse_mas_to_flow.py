import json

# Đường dẫn file JSON đầu vào
for k in range(0, 150):
    input_file = f'workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v3_test/mas/mas_{k}.json'  # file chứa mảng subtask, ví dụ: ["subtask_1", "subtask_2", ...]

    # Đường dẫn file output chứa cấu trúc workflow
    output_file = f'workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v3_test/flow/flow_{k}.json'

    # Đọc mảng subtasks từ file JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        subtasks = json.load(f)  # Giả sử đây là một list các tên subtask

    # Ghi ra file theo format yêu cầu
    res = '[sequential]\n'
    for subtask in subtasks:
        res += f'    [{subtask['subtask_id']}]\n'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)

    print(f"Đã tạo workflow tại: {output_file}")
