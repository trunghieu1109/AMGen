import json
import csv

def jsonl_to_csv(jsonl_file, csv_file):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    # Gộp tất cả keys lại để đảm bảo header đầy đủ
    all_keys = set()
    for entry in lines:
        all_keys.update(entry.keys())
    all_keys = sorted(all_keys)  # Đảm bảo thứ tự cột cố định

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for entry in lines:
            writer.writerow(entry)

# Example usage
jsonl_to_csv('data/hotpotqa_validate.jsonl', 'dataset/hotpotqa_validate.csv')
