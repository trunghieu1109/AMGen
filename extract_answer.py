import os
cnt = 0
def read_matching_files(base_path):
    result = {}
    
    # Lặp qua các folder trong thư mục gốc
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.isdir(folder_path):
            # Tìm các file thỏa điều kiện trong folder hiện tại
            for file_name in os.listdir(folder_path):
                if file_name.startswith("gpt-4.1-mini") and file_name.endswith("plan_judge"):
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Đọc nội dung file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    
                    splitted_text = content.split("Question:")[-2:]
                    print(len(splitted_text))
                    
                    # Lưu vào dict theo tên folder
                    result[folder_name] = splitted_text
                    break  # Nếu chỉ cần một file đầu tiên mỗi folder

    return result

# Ví dụ sử dụng
base_path = "results/dev_31_refactored/question/meta_agent/workflow_search/gpqa_diamond"  # Thay bằng đường dẫn thật
contents = read_matching_files(base_path)

for file_name, content in contents.items():
    with open("output_gpqa_gpt-4.1-mini.txt", 'a+', encoding='utf-8') as f:
        f.write(f"\n====================={file_name}=====================\n")
        f.write("\n".join(content))


print(len(contents.values()))