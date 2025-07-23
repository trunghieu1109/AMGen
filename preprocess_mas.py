import os
import json

def clean_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for i in range(len(data)):
                    data[i].pop('abstracted_objective')
                    data[i].pop('subtask_name')

                # Xoá 2 field nếu có
                

                # Ghi ngược lại vào file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"✅ Processed: {filename}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

# Ví dụ sử dụng
clean_json_files("merged_mas/workflow_analysis-gpt-4o-mini-o4-mini_v8-drop_v3_test/mas")
