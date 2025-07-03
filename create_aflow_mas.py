import os
from pathlib import Path
import json

folder_path = 'AFlow/results/results'
datasets = [name for name in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, name))]

for dataset in datasets:
    aflow_mas = []
    print(dataset)
    graph_folder_path = f'{folder_path}/{dataset}/graphs'
    round_folders = [name for name in os.listdir(graph_folder_path)
               if os.path.isdir(os.path.join(graph_folder_path, name))]
    
    with open(f'data/{dataset.lower()}_validate.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    sample_problems = []
    for dat in data[:5]:
        problem = ""
        if 'context' in dat:
            problem += f"Context: {dat['context']}\n"
            
        if 'question' in dat:
            problem += f"Question: {dat['question']}\n"
            
        if 'prompt' in dat:
            problem += f"Prompt: {dat['prompt']}\n"
            
        if 'entry_point' in dat:
            problem += f"Entrypoint: {dat['entry_point']}\n"
            
        if 'problem' in dat:
            problem += f"Problem: {dat['problem']}\n"
            
        sample_problems.append(problem)
    
    max_score = 0
    for round in round_folders:
        round_folder_path = f"{graph_folder_path}/{round}"
        # print(round_folder_path)
        round_folder_path = Path(round_folder_path)
        csv_files = round_folder_path.glob('*.csv')

        for file in csv_files:
            last_ = str(file).rfind('/')
            csv_file_name = str(file)[last_ + 1:]
            
            score_pos = csv_file_name.find("_")
            # print(csv_file_name[:score_pos])
            score = csv_file_name[:score_pos]
            max_score = max(max_score, float(score))
    print("Dataset: ", dataset)
    print("Max Score: ", max_score)
    
    print("Range: ", f"[{max_score * 0.8}, {max_score}]")
    qualified_round = []
    for round in round_folders:
        round_folder_path = f"{graph_folder_path}/{round}"
        # print(round_folder_path)
        round_folder_path = Path(round_folder_path)
        csv_files = round_folder_path.glob('*.csv')

        max_inner = 0
        for file in csv_files:
            last_ = str(file).rfind('/')
            csv_file_name = str(file)[last_ + 1:]
            
            score_pos = csv_file_name.find("_")
            # print(csv_file_name[:score_pos])
            score = csv_file_name[:score_pos]
            max_inner = max(max_inner, float(score))                             
            
        if max_inner > max_score * 0.9:
            # print(round_folder_path)
            qualified_round.append(round_folder_path)

            with open(str(round_folder_path) + "/graph.py", 'r', encoding='utf-8') as f:
                graph = f.read()

            start_workflow = graph.find("async def __call__")
            aflow_mas.append({
                "problem": sample_problems,
                "code": graph[start_workflow:],
                'iteration': 0
            })
            # print(graph[start_workflow:])
            
    with open(f'aflow_{dataset}.json', 'w', encoding='utf-8') as f:
        json.dump(aflow_mas, f, ensure_ascii=False, indent=4)


print(sample_problems)
