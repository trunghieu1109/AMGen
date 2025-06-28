import json

with open("workflow_analysis-gpt-4o-mini-o4-mini_v8-aime24/abstracted_workflow/abstract_workflow_description.json", "r") as f:
    data = json.load(f)

mas_chain = []
for dat in data:
    mas_chain.append(dat['chain'])
    
with open("workflow_analysis-gpt-4o-mini-o4-mini_v8-aime24/abstracted_workflow/workflow_chains.json", "w") as f:
    json.dump(mas_chain, f, indent=2)  # in