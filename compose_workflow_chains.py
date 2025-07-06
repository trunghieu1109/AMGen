import json

def extract_chain(input_file, output_file):
    # Bước 1: Đọc JSON từ file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # data là list các dict

    # Bước 2: Trích xuất field "chain"
    chains = [item['chain'] for item in data if 'chain' in item]

    # Bước 3: Ghi kết quả ra file JSON mới
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chains, f, ensure_ascii=False, indent=4)

# Ví dụ sử dụng
extract_chain('workflow_analysis-gpt-4o-mini-o4-mini_v8-drop_v3/abstracted_workflow/abstract_workflow_description.json', 'workflow_analysis-gpt-4o-mini-o4-mini_v8-drop_v3/abstracted_workflow/workflow_chains.json')
