import argparse
import copy
import json
import os
import random
from collections import namedtuple
import concurrent.futures
from tqdm.asyncio import tqdm_asyncio
import traceback
import time

import backoff
import numpy as np
import openai
from tqdm import tqdm
import types

import re

from typing import Any
from datasets import load_dataset
from prompts.abstract_based_prompt import INTERACTION_PATTERN, ABSTRACTED_WORKFLOW_TEMPLATE
import pandas as pd
import common
from common import HTML_JINJA, get_init_archive, get_prompt, get_reflexion_prompt, SingleEvalResult, get_reflexion_after_eval
from common import get_json_response_from_gpt, get_json_response_from_gpt_reflect, _pack_message
from utils import random_id, bootstrap_confidence_interval
from common import ANSWER_PATTERN, shorten_context, merge_context
from collections import Counter
import copy
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from utils import  extract_xml
from shared_vars import set_global, get_global
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize
from Levenshtein import distance
import pickle as pkl
from abstract_v2 import MASAbstraction, levenshtein_array_to_array, DESCRIPTION_FOR_OPERATOR
import asyncio

import operator_templates
from agent_system import AgentSystem
from llm_agent_base import LLMAgentBase, Info

ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

async def evaluate_forward_fn(args, example_id, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py

    # print('forward_str: ',forward_str)

    # if you want debug, remove the section so that you can see the detailed error line
    namespace = {}
    try:
        safe_forward_str = forward_str.replace("\\", "\\\\")
        exec(safe_forward_str, globals(), namespace)
        print("Safe forward function: ", safe_forward_str)
    except Exception as e:
        print("❌ Lỗi khi thực thi forward_str:", str(e))
        error_trace = traceback.format_exc()
        print("Full error trace:\n", error_trace)
        raise RuntimeError(f"exec() failed with error: {e}") from e
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, f"forward_{example_id}", func)

    agentSystem = AgentSystem()

    global_max_workers = get_global("global_max_workers")
    global_task_queue = get_global("global_task_queue")
    global_task_queue = global_task_queue[str(example_id)]
    
    global_answers = get_global("global_answers")
    global_answers = global_answers[str(example_id)]

    agentSystem.node_model = get_global("global_node_model")
    agentSystem.cot_instruction = get_global("global_cot_instruction")
    agentSystem.max_sc = get_global("global_max_sc")
    agentSystem.max_round = get_global("global_max_round")
    agentSystem.debate_role = get_global("global_debate_role")
    agentSystem.dataset = get_global("global_dataset")
    agentSystem.example_id = example_id
    agentSystem.instance_id = get_global("global_instance_id")
    
    results = None

    print("Global Task Queue: ", global_task_queue)
        
    async def call_forward(example_id, task):
        method_name = f"forward_{example_id}"
        forward_fn = getattr(agentSystem, method_name)
        return await forward_fn(task)
    start_time = time.time()
    results, logs = await call_forward(example_id, global_task_queue[0])
    end_time = time.time()
    total_time = end_time - start_time
    results = [results]

    prompt_messages = [res.prompt for q_idx, res in enumerate(results)]
    response_texts = [str(res.content) for q_idx, res in enumerate(results)]
    if not get_global("global_no_decompose"):
        sub_tasks = [res.sub_tasks for q_idx, res in enumerate(results)]
        sub_tasks_text = sub_tasks[0] # only one sample
    else:
        sub_tasks = None
        sub_tasks_text = None

    agents = [res.agents for q_idx, res in enumerate(results)]

    print('response_texts: ',response_texts[0])
    print('gold answers: ',global_answers[0])
    print('length: ', len(response_texts), len(global_answers))

    global_score_compute = get_global("global_score_compute")
    global_example_id = example_id
    global_n = get_global("global_n")
    global_n = global_n[str(example_id)]
    
    global_questions = get_global("global_questions")
    global_questions = global_questions[str(example_id)]
    global_answers = get_global("global_answers")
    global_answers = global_answers[str(example_id)]
    global_use_oracle_verifier = get_global("global_use_oracle_verifier")
    global_judge_path = get_global("global_judge_path")
    global_judge_path = global_judge_path[str(example_id)]
    
    global_instance_id = get_global("global_instance_id")
    global_code_snippet = get_global("global_code_snippet")

    result_list = [
        await global_score_compute(
            global_example_id,
            global_n,
            prompt_messages[response_text_id], 
            global_questions[response_text_id], 
            response_text, 
            global_answers[response_text_id],
            sub_tasks_text, 
            global_use_oracle_verifier,
            global_judge_path, 
            global_instance_id,
            global_code_snippet) 
        for response_text_id, response_text in enumerate(response_texts)
        ]
    
    raw_results = results
    
    acc_oracle_verifier_list = [x[0] for x in result_list]
    acc_model_verifier_list = [x[1] for x in result_list]
    result_list = [x[2] for x in result_list]
    results = common.aggregate_results(result_list)

    print(f"acc_oracle_verifier_list:", acc_oracle_verifier_list)
    print(f"acc_model_verifier_list:", acc_model_verifier_list)


    return acc_oracle_verifier_list, acc_model_verifier_list, results, sub_tasks, agents, response_texts, raw_results, logs, response_texts[0], global_answers[0], total_time
            
async def apply_abstract_workflow_enhance(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None, date_time="", save_file="", abstract_mas_path="", specific_op_desc={}):

    # if example_id != 160:
    #     return 0, 0, 0, ""

    start_time_ = time.time()
    total_execution_time = 0
    
    global_node_model = get_global("global_node_model")
    
    # declare results path
    result_path = expr_name + f"{args.dataset}"
    expr_name = expr_name + f"{args.dataset}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"
    oracle_acc_result_path = f'results/{args.dataset}/{save_file}/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_path = Path(oracle_acc_result_path)
    oracle_acc_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_judge")
    os.makedirs(os.path.dirname(judge_path), exist_ok=True)
    global_judge_path = get_global('global_judge_path')
    global_judge_path[str(example_id)] = judge_path
    final_results_path = f'results/{args.dataset}/{save_file}/{meta_model}_{global_node_model}/final_results_{example_id}.json'
    result_path = Path(final_results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(f"logs/{args.dataset}/{meta_model}", exist_ok=True)
    log_path = f"logs/{args.dataset}/{meta_model}/{date_time}_logs_{example_id}.txt"
    
    global_ns = []
    global_use_oracle_verifier = get_global("global_use_oracle_verifier")
    output_description = get_global("global_output_description")
    acc_oracle_verifier_list = [0]
    total_time = 0
    
    final_results = []
    max_score = 0
    logs = []
    evaluation = []
    task_decomposition = None
    next_solution = {
        'code': ""
    }
    
    verifier_hub = [
        'o4-mini',
    ]
    
    operator_list = {pattern: {'description': specific_op_desc[pattern]['description'], 'characteristics': specific_op_desc[pattern]['unique_characteristics']} for pattern in specific_op_desc}

    max_workers = 1
    set_global("global_max_workers", max_workers)
    
    # load mas chains list from offline abstract workflows
    with open(f'{abstract_mas_path}/abstracted_workflow/workflow_chains.json', 'r', encoding='utf-8') as f:
        default_mas_chain = json.load(f)
    
    # load mapping from cluster to subtask
    with open(f'{abstract_mas_path}/cluster_to_subtask_mapping.json', 'r', encoding='utf-8') as f:
        cluster_to_subtask = json.load(f)
        
    abstracted_subtask = [subtask for name, subtask in cluster_to_subtask.items()]
    
    # load subtask name from offline mas abstraction
    subtask_names = set()
    for aw in abstract_workflow:
        # print(aw['flow'])
        for stage_id, stage in aw['flow'].items():
            if 'Title' in stage:
                if isinstance(stage['Title'], list):
                    for sub_title in stage['Title']:
                        subtask_names.add(sub_title)
                else:
                    subtask_names.add(stage['Title'])
                    
    for name in subtask_names:
        print(name)
        
    # config task_queue with specific datasets
    if 'gpqa_diamond' in args.dataset:
        task_queue = [Info(field_name, author, {"question": content.question, "choice1": content.choice1, "choice2": content.choice2, "choice3": content.choice3, "choice4": content.choice4}, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]
    else:
        task_queue = [Info(field_name, author, content, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]

    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue    
    set_global("global_task_queue", global_task_queue)
    
    # get sample interaction pattern
    interaction_pattern = INTERACTION_PATTERN
    
    '''
    ================================================== CHOOSE SUITABLE ABSTRACTED WORKFLOW ======================================================
    '''
    
    # decompose task to high-level subtasks
    high_level_decomposition = await operator_templates.high_level_task_decomposition(meta_model, task_queue[0].content, 3)
    # high_level_decomposition = [{'objective': 'Understand the Michael addition mechanism and the role of nucleophiles and electrophiles in forming products'}, {'objective': "Analyze each reaction's reactants and conditions to predict the major product formed via Michael addition"}, {'objective': 'Determine the structures of intermediates and resonance-stabilized species involved in each reaction'}, {'objective': 'Compare predicted products with provided answer choices to select the correct products for reactions A, B, and C'}]
    print("\n========================= high_level_decomposition ==========================\n ", high_level_decomposition)
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "high level task decomposition"
        content = high_level_decomposition
        
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
        
    # return 1, 1, 1, ""

    high_level_subtasks = [subtask['objective'] for subtask in high_level_decomposition]
    
    # embeddings, postprocess and clustering the embeddings
    abstractor = MASAbstraction()
    abstracted_subtasks = await abstractor.abstract_task_decomposition(task_queue[0].content, high_level_subtasks, subtask_names, abstracted_subtask)
    merged_subtasks = [f"{a_subtask['subtask_name']}: {a_subtask['abstracted_objective']}" for a_subtask in abstracted_subtasks]
    print("\n=============================== High-level Abstracted MAS ============================\n", merged_subtasks)
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "abstracted high level task decomposition"
        content = merged_subtasks
        
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    # return 1, 1, 1, ""
    embeddings = await abstractor.embedding_subtask(merged_subtasks)
    
    # get kmeans clustering object from former abstraction
    with open(f'{abstract_mas_path}/kmeans.pkl', 'rb') as f:
        kmeans = pkl.load(f)
    
    # get pca object from former abstraction, which fit with available data
    with open(f'{abstract_mas_path}/pca.pkl', 'rb') as f:
        pca = pkl.load(f)
        
    normalized_embeddings = normalize(embeddings, norm="l2")
    reduced_embeddings = pca.transform(normalized_embeddings)
    
    cluster_id = kmeans.predict(reduced_embeddings)
    mas_chain = []
    for num_id, idx in enumerate(cluster_id):
        if str(idx) not in mas_chain or num_id == len(cluster_id) - 1:
            mas_chain.append(str(idx))

    # filter sequential only mas
    sequential_only_mas = []
    loop_contain_mas = []
    conditional_contain_mas = []
    
    for idx, mas in enumerate(default_mas_chain):
        contain_loop = False
        contain_conditional = False
        if 'start_conditional' in mas:
            conditional_contain_mas.append({
                'id': idx,
                'mas': mas
            })
            contain_conditional = True
        
        if 'start_loop' in mas:
            loop_contain_mas.append({
                'id': idx,
                'mas': mas
            })
            contain_loop = True
        
        if not contain_conditional and not contain_loop:
            sequential_only_mas.append({
                'id': idx,
                'mas': mas
            })            

    async def get_no_flow_mas(mas):
        mas_chain_no_flow = []
        for stage in mas:
            if isinstance(stage, str) and not stage.isdigit():
                continue
            
            mas_chain_no_flow.append(stage)
        return mas_chain_no_flow
            
    async def choose_the_most_similar_mas(mas_chain, mas_chain_list, abstract_workflow_, distance_threshold=2):
        if len(mas_chain_list) == 0:
            return None
        
        mas_chain_no_flow_list = []
        for mas in mas_chain_list:
            mas_chain_no_flow = await get_no_flow_mas(mas['mas'])
                
            mas_chain_no_flow_list.append({
                "id": mas['id'],
                "mas": mas_chain_no_flow
            })
        
        distance_mas_pairs = [
            (levenshtein_array_to_array(mas_chain, mas['mas']), mas)
            for mas in mas_chain_no_flow_list if len(mas['mas']) > 0
        ]

        min_distance = min(distance_mas_pairs, key=lambda x: x[0])[0]

        min_distance_candidates = [
            mas for dist, mas in distance_mas_pairs if dist == min_distance
        ]

        if min_distance > distance_threshold:
            print("Cannot find the suitable MAS")
            return None

        sorted_chains = sorted(min_distance_candidates, key=lambda mas: -len(mas['mas']))

        # Lấy 2 chuỗi giống nhất
        closest_2 = sorted_chains[:1]
        print("MAS Chain: ", closest_2[0]['mas'], " Id: ", closest_2[0]['id'])
        print("Origin mas chain: ", mas_chain)
        for idx, cls_ in enumerate(closest_2):
            print(f"Abstract mas {cls_['id']}")
            print(f"Levenshtein distance {idx}: ", levenshtein_array_to_array(mas_chain, cls_['mas']))
        
        # get 2 the most similar workflow
        workflow_index = []
        for idx, awd in enumerate(abstract_workflow):
            awd_chain = await get_no_flow_mas(awd['chain'])
            if str(awd_chain) == str(closest_2[0]['mas']):
                workflow_index.append(closest_2[0]['id'])

        if len(workflow_index) == 0:
            workflow_index = [mas_chain_no_flow_list[random.randint(0, len(mas_chain_no_flow_list) - 1)]['id']]

        print("workflow index: ", workflow_index)
        # return 1, 1, 1, ""
        
        filtered_workflow = [abstract_workflow[idx] for idx in workflow_index]
        
        print("Filtered workflow: ", [fw['name'] for fw in filtered_workflow])
        
        return filtered_workflow

    # choose the most similar sequential only mas
    filtered_sequential_workflow = await choose_the_most_similar_mas(mas_chain, sequential_only_mas, abstract_workflow)
    if filtered_sequential_workflow:
        with open(log_path, "a+", encoding="utf-8") as f:
            
            phase = "Sequential Abstract MAS choosing"
            content = f"Query-based chain: {mas_chain}\nSimilar chain: {filtered_sequential_workflow[0]['chain']}\nLevenshtein distance: {levenshtein_array_to_array(mas_chain, filtered_sequential_workflow[0]['chain'])}"
            
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
    filtered_loop_workflow = await choose_the_most_similar_mas(mas_chain, loop_contain_mas, abstract_workflow)
    if filtered_loop_workflow:
        with open(log_path, "a+", encoding="utf-8") as f:
        
            phase = "Loop Abstract MAS choosing"
            content = f"Query-based chain: {mas_chain}\nSimilar chain: {filtered_loop_workflow[0]['chain']}\nLevenshtein distance: {levenshtein_array_to_array(mas_chain, filtered_loop_workflow[0]['chain'])}"
            
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
        
    # print("Filtered loop chain", filtered_loop_workflow[0]['chain'])
    # print("Filtered sequential chain", filtered_sequential_workflow[0]['chain'])
    
    task_content = task_queue[0].content
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, task_content, task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    
    print(f'============ Initial Archive: Test New Architecture =================')
    default_global_n = get_global("global_n")
    default_global_n[str(example_id)] = f"Test New Architecture_{example_id}"
    set_global("global_n", default_global_n)

    global_n = get_global("global_n")
    global_n = global_n[str(example_id)]
    global_ns.append(global_n)
    
    task_detail_analysis = await operator_templates.task_analysis(meta_model, task_queue[0].content)
    print(f"\n================== Query analysis: {task_detail_analysis}========================\n")
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, str(task_content) + "\n\nDetailed Analysis: \n" + str(task_detail_analysis), task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "Task detailed analysis"
        content = task_detail_analysis
        
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    # return 1, 1, 1, ""
    
    '''
    ================================================== CONCRETIZE WORKFLOW ======================================================
    '''
    
    # load chosen abstracted workflow for each type
    sequential_only_aw = None
    if filtered_sequential_workflow:
        with open(filtered_sequential_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
            sequential_only_aw_ = json.load(f)
        sequential_only_aw = {
            'flow': filtered_sequential_workflow[0]['flow'],
            'specific_workflow': sequential_only_aw_
        }
            
    loop_contain_aw = None
    if filtered_loop_workflow:
        with open(filtered_loop_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
            loop_contain_aw_ = json.load(f)
        
        loop_contain_aw = {
           'flow': filtered_loop_workflow[0]['flow'],
           'specific_workflow': loop_contain_aw_
        }
    
    merge_filtered_aw = await operator_templates.merge_filtered_workflow(meta_model, sequential_only_aw, loop_contain_aw)
    # merge_filtered_aw = {'Control Flow 0': {'flow_type': 'start sequential', 'flow_desc': 'Start a sequential code flow, with multiple stages performed in turn'}, 'Stage 0': {'stage_id': 'stage_0', 'stage_name': 'extract_and_categorize_information', 'abstracted_objective': 'Analyze an input to identify, extract, and categorize its relevant elements, attributes, criteria, constraints, and relationships to support subsequent processing or reasoning.', 'agent_collaboration': ['CoT', 'SC_CoT'], 'dependencies': []}, 'Stage 1': {'stage_id': 'stage_1', 'stage_name': 'analyze_relationships', 'abstracted_objective': 'Analyze and characterize the relationships, interactions, or transformations among given inputs to determine their functional associations, dependencies, or resulting outcomes according to specified criteria.', 'agent_collaboration': ['CoT', 'SC_CoT'], 'dependencies': ['stage_0']}, 'Control Flow 1': {'flow_type': 'start loop', 'flow_desc': 'Start a loop code flow, with multiple stages performed iteratively'}, 'Stage 2': {'stage_id': 'stage_2', 'stage_name': ['construct_intermediate_steps', 'refine_output', 'derive_transformed_output', 'validate_entity'], 'abstracted_objective': 'Generate a structured sequence of intermediate steps by applying a systematic procedure to given inputs, progressively transforming them to produce an initial or provisional output along with any necessary reasoning or documentation, then transform these preliminary outputs by simplifying, consolidating, and enhancing them to produce a refined final result that satisfies defined criteria or constraints, and evaluate an entity against predefined criteria to determine its compliance, correctness, consistency, and validity, producing an assessment outcome or feedback.', 'agent_collaboration': ['CoT', 'SC_CoT'], 'dependencies': ['stage_0', 'stage_1']}, 'Control Flow 2': {'flow_type': 'end loop', 'flow_desc': 'End of current loop code flow'}, 'Stage 3': {'stage_id': 'stage_3', 'stage_name': 'select_best_candidate', 'abstracted_objective': 'Evaluate a collection of candidate elements against defined criteria or conditions and select the element or subset that best satisfies the specified requirements.', 'agent_collaboration': ['CoT', 'SC_CoT', 'Review', 'Debate', 'Aggregate', 'AnswerGenerate', 'Reflexion'], 'dependencies': ['stage_0', 'stage_1', 'stage_2']}, 'Control Flow 3': {'flow_type': 'end sequential', 'flow_desc': 'End of current sequential code flow'}}
    # return 1, 1, 1, ""
    # dependencies_and_agent_collaboration = []
    # for idx, st in enumerate(sequential_only_aw['specific_workflow']):
    #     if 'dependencies' in st:
    #         dependencies_and_agent_collaboration.append({
    #             'stage_id': st['subtask_id'].replace('subtask', 'stage'),
    #             'agent_collaboration': st['agent_collaboration'],
    #             'dependencies': [f"From {st['subtask_id'].replace('subtask', 'stage')} to stage_{dep.replace("subtask", "stage")}" for dep in st['dependencies']]
    #         })
    
    # print(dependencies_and_agent_collaboration)
    
    # print(task_queue_tmp[0].content)
    # print(dependencies_and_agent_collaboration)
    # print(sequential_only_aw['flow'])
    
    # decompose query into multiple subtasks
    # task_decomposition = await operator_templates.specific_task_decomposition(meta_model, task_queue_tmp[0].content, dependencies_and_agent_collaboration, aw_flow=sequential_only_aw['flow'], potential_op=operator_list)
    task_decomposition = await operator_templates.specific_task_decomposition(meta_model, task_queue_tmp[0].content, aw_flow=merge_filtered_aw, potential_op=operator_list)
    print("\n============= Task Decomposition: =============\n")
    print(json.dumps(task_decomposition, indent=4, ensure_ascii=False))
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "Task Decomposition"
        content = task_decomposition
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    # return 1, 1, 1, ""
    stage_desc = str(merge_filtered_aw).replace("subtask", "stage")
    
    # generate new workflow that concretized for this query
    next_solution = await operator_templates.generate_concretized_workflow(meta_model, task_decomposition, interaction_pattern, task_queue_tmp[0].content, potential_op=specific_op_desc)
    print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
    # return 1, 1, 1, ""
    # evaluate multi agent system
    try:
        next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
        acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"])
        total_execution_time += total_time
    except Exception as e:
        print("Error: ", str(e))
        error_trace = traceback.format_exc()
        print("Full error trace:\n", error_trace)
        with open(f"error_mas/error_mas_{example_id}.py", "w") as f:
            f.write(next_solution['code'])
        return -1, -1, -1, ""
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "Concretized MAS"
        content = next_solution['code']
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    # return 1, 1, 1, ""
    
    # save results
    if global_use_oracle_verifier:
        acc_list = acc_oracle_verifier_list
    else:
        acc_list = acc_model_verifier_list

    print(f"acc_list:", acc_list)
    print(f"mean acc_list:", np.mean(acc_list))
    
    extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

    if '[TOO_HARD]' in extracted_answer:
        extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        

    converted_code_filename = os.path.join(args.save_dir, f'{expr_name}_Test New Architecture_{example_id}_first_gen_converted.py')
    with open(converted_code_filename, "w") as fh:
        fh.write(next_solution['code'])    
    
    print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))

    with open(oracle_acc_result_path, "a+") as fh:
        fh.write(f'experiemnt {example_id}: 1 (initial Test New Architecture): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')
    
    max_score = max(max_score, acc_oracle_verifier_list[0])
    final_results.append({
        "example_id": example_id,
        "score": max_score,
        "max_cost": get_global("global_COST_TOTAL")
    })
    # return 1, 1, 1, ""
    '''
    ================================================== REFINE WORKFLOW ======================================================
    '''
    
    # if the first concretized workflow is not optimal, try to evaluate and refine it
    if int(max_score) == 0:
        subtask_desc = logs
            
        # get evaluation from many llm experts
        evaluation = await operator_templates.evaluate_workflow(task_queue_tmp[0].content, subtask_desc, current_ans, output_description, verifier_hub)
        # evaluation = [{'failure_reason': "The previous reasoning process consistently selected choice 2 based on mechanistic and nomenclature analysis, but the professor's feedback indicates this final answer is incorrect. The failure stems from a subtle but critical misinterpretation of the product structures, specifically the identity and tautomeric form of compound C and the oxidation state of substituents in product B. The agents incorrectly assumed the hydroxy form for product B and cyclohexane-1,3-dione (not its hydroxy derivative) for compound C without fully considering alternative tautomeric or structural possibilities that align better with the given product names and reaction conditions. This led to a systematic but flawed conclusion favoring choice 2.", 'feedback': 'The detailed reasoning by previous agents correctly identified the general Michael addition mechanisms and the nucleophilic attack at the β-carbon. However, the critical error lies in the assumptions about the tautomeric forms and oxidation states of the products, especially for reactions (B) and (C). The agents assumed that acidic workup in reaction (B) necessarily leads to a hydroxy-substituted product, neglecting that the product name in some choices indicates an oxo substituent, which might be more consistent with the reaction conditions or product stability. Similarly, for reaction (C), the agents favored cyclohexane-1,3-dione as compound C, dismissing the hydroxy derivative, but the product name and reaction conditions might better support the hydroxycyclohexane-1,3-dione form. This misinterpretation caused the agents to select choice 2, which mismatches the actual product identities. The error originated in the subtask analyses where tautomeric equilibria and oxidation states were not rigorously evaluated or alternative plausible structures were insufficiently considered. The context provided was adequate for general mechanism analysis but lacked explicit structural clarifications or spectral/experimental data that could help distinguish between hydroxy and oxo forms. The collaboration pattern (SC_CoT and Debate) was effective for consensus but did not enforce critical evaluation of tautomeric possibilities or alternative product forms, leading to confirmation bias towards choice 2.', 'suggestion': '1. Refine subtasks to explicitly include evaluation of tautomeric forms and oxidation states of products, especially when acidic or basic workup is involved. This can be done by adding a dedicated subtask focusing on tautomerism and oxidation state analysis, supported by chemical logic and nomenclature conventions.\n\n2. Enhance context passing between subtasks by including detailed structural information, possible tautomeric forms, and their expected stability under given reaction conditions. This can help subsequent agents critically assess product identities rather than relying on assumptions.\n\n3. Upgrade collaboration patterns for critical subtasks (e.g., product identification and validation) from SC_CoT to Reflexion or Debate with explicit prompts to challenge assumptions and consider alternative structures. This will reduce confirmation bias and improve robustness of conclusions.\n\nImplementing these steps will address the root cause of the failure by ensuring that tautomeric and oxidation state considerations are systematically analyzed and that agents critically evaluate all plausible product forms before finalizing answers.'}]
        with open(log_path, "a+", encoding="utf-8") as f:
        
            phase = "Evaluation from verifiers"
            content = evaluation
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
        print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
        # refined_task_decomposition = await operator_templates.specific_task_decomposition(meta_model, task_queue_tmp[0].content, dependencies_and_agent_collaboration, aw_flow=sequential_only_aw['flow'], evaluation=evaluation, prev_task_decomposition=task_decomposition, is_refinement=True, potential_op=specific_op_desc)
        refined_task_decomposition = await operator_templates.specific_task_decomposition(meta_model, task_queue_tmp[0].content, aw_flow=merge_filtered_aw, evaluation=evaluation, prev_task_decomposition=task_decomposition, is_refinement=True, potential_op=specific_op_desc)
        
        print("\n============= Refined Task Decomposition: =============\n", refined_task_decomposition)
        # return 1, 1, 1, ""
        with open(log_path, "a+", encoding="utf-8") as f:
            
            phase = "Refined Task Decomposition"
            content = refined_task_decomposition
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
        # regenerate workflow
        next_solution = await operator_templates.generate_concretized_workflow(meta_model, refined_task_decomposition, interaction_pattern, task_queue_tmp[0].content)
        print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
        
        # re-evaluate refined workflow
        try:
            next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
            acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"])
            total_execution_time += total_time
        except Exception as e:
            print("Error: ", str(e))
            error_trace = traceback.format_exc()
            print("Full error trace:\n", error_trace)
        with open(log_path, "a+", encoding="utf-8") as f:
        
            phase = "Refined MAS"
            content = next_solution['code']
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
        # save refined results
        judge_path = os.path.join(args.save_dir, f"{expr_name}_Test New Architecture_{example_id}_full_response_refined")
        with open(judge_path, 'w') as judge_file:
            judge_file.write(f'Question: {task_queue[0].content}\nIteration: Test New Architecture\nFull Response:{raw_results}')

        if global_use_oracle_verifier:
            acc_list = acc_oracle_verifier_list
        else:
            acc_list = acc_model_verifier_list
            
        print(f"acc_list:", acc_list)
        print(f"mean acc_list:", np.mean(acc_list))
       
        extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

        if '[TOO_HARD]' in extracted_answer:
            extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        
        # save results

        converted_code_filename = os.path.join(args.save_dir, f'{expr_name}_Test New Architecture_{example_id}_converted_refined.py')
        with open(converted_code_filename, "w") as fh:
            fh.write(next_solution['code'])    

        print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))

        with open(oracle_acc_result_path, "a+") as fh:
            fh.write(f'experiemnt {example_id}: 1 (initial Test New Architecture): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')
        
        max_score = max(max_score, acc_oracle_verifier_list[0])
    
    # logs results to console
    print("score: ", acc_oracle_verifier_list[0])
    print("Current answer: ", current_ans)
    print("ground truth: ", ground_truth)
                
    end_time_ = time.time()
    total_time = end_time_ - start_time_        
    
    final_results.append({
        "example_id": example_id,
        "score": max_score,
        "total_time": total_time,
        "total_execution_time": total_execution_time,
        "max_cost": get_global("global_COST_TOTAL"),
        "max_execution_cost": get_global("global_COST_EXECUTION")
    })
    
    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
            
    return acc_oracle_verifier_list[0], total_time, total_execution_time, ""

async def recheck_mas(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None):

    if example_id != 166:
        return 0, 0, 0, ""

    start_time_ = time.time()
    total_execution_time = 0
    total_execution_cost = 0
    questions = get_global("global_questions")
    questions = questions[str(example_id)]
    global_node_model = get_global("global_node_model")

    print(f"problem length: {len(questions)}")
    max_workers = min(len(questions), args.max_workers) if args.multiprocessing else 1
    
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/abstracted_workflow/workflow_chains.json', 'r', encoding='utf-8') as f:
        default_mas_chain = json.load(f)
    result_path = expr_name + f"{args.dataset}"
    expr_name = expr_name + f"{args.dataset}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"

    if 'gpqa_diamond' in args.dataset:
        task_queue = [Info(field_name, author, {"question": content.question, "choice1": content.choice1, "choice2": content.choice2, "choice3": content.choice3, "choice4": content.choice4}, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]
    else:
        task_queue = [Info(field_name, author, content, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]

    set_global("global_max_workers", max_workers)
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue    
    set_global("global_task_queue", global_task_queue)

    next_solution_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_next_solution.json")
    msg_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_msg.json")
    mem_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_mem.json")
    file_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_archive.json")
    result_path = f'results/{args.dataset}/abstract_workflow/{meta_model}_{global_node_model}_{verifier_model}.results'
    oracle_acc_result_path = f'results/{args.dataset}/recheck_mas/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_result_path = f'results/{args.dataset}/recheck_mas/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_path = Path(oracle_acc_result_path)
    oracle_acc_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_acc_path = Path(result_path)
    result_acc_path.parent.mkdir(parents=True, exist_ok=True)

    judge_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_judge")
    reponse_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_reponse")
    os.makedirs(os.path.dirname(judge_path), exist_ok=True)

    print('file_path: ',file_path)
    print('msg_path: ',msg_path)
    print('result_path: ',result_path)
    print('next_solution_path: ',next_solution_path)
    print('oracle_acc_result_path: ',oracle_acc_result_path)
    print('judge_path: ',judge_path)
    print('reponse_path: ',reponse_path)
    print('mem_path: ',mem_path)

    global_judge_path = get_global('global_judge_path')
    global_judge_path[str(example_id)] = judge_path

    set_global("global_judge_path", global_judge_path)
    
    global_reponse_path = get_global('global_reponse_path')
    global_reponse_path[str(example_id)] = reponse_path

    set_global("global_reponse_path", global_reponse_path)

    if os.path.exists(mem_path):
        with open(mem_path, 'r') as json_file:
            memory = json.load(json_file)
    else:
        memory = []

    if os.path.exists(reponse_path):
        with open(reponse_path, 'r') as json_file:
            global_response = json.load(json_file)
            
        global_response_dict = get_global('global_response_dict')
        global_response_dict[str(example_id)] = global_response
        
        set_global("global_response_dict", global_response_dict)

    global_use_oracle_verifier = get_global("global_use_oracle_verifier")

    global_ns = []
    detailed_analysis = '''
1. Extract and Summarize Given Information:
- The quantum state |psi> is defined as a superposition: |psi> = (cos(phi)|alpha> + sin(phi)|-alpha>)/N.
- Parameters: alpha (amplitude), phi (phase), and N (normalization constant).
- Normalization constant: N = sqrt(1 + sin(2*phi)*exp(-2*alpha^2)).
- The measure of non-Gaussianity (nG) is given by the relative entropy difference: del_b = trace(rho ln rho) - trace(tau ln tau), where rho is the density matrix of the non-Gaussian state |psi><psi|, and tau is the density matrix of a reference Gaussian state.
- Specific values for calculation: phi = -π/4, alpha = 0.5.

2. Analyze Relationships Between Components:
- The state |psi> is a linear combination of coherent states |alpha> and |-alpha>, weighted by trigonometric functions of phi and normalized by N.
- The normalization constant N depends on both phi and alpha, ensuring |psi> is a valid quantum state.
- The relative entropy measure compares the entropy of the non-Gaussian state rho to that of the Gaussian reference tau, quantifying the deviation from Gaussianity.
- The choice of phi and alpha fixes the specific form of the state and thus the density matrix rho.
- The problem implicitly requires constructing rho and tau, computing their logarithms and traces, which are linked through quantum information theory.

3. Identify the Field of Study:
- The problem lies in quantum physics, specifically quantum information theory.
- Subfields include quantum optics (coherent states, Schrödinger cat states), quantum state characterization, and quantum entropy measures.
- Mathematical tools involve linear algebra (density matrices), functional analysis (trace and logarithm of operators), and probability theory (entropy).
- Applications include quantum computing, quantum communication, and studies of quantum non-classicality.

4. Highlight Aspects Needing Clarification:
- The explicit form or construction of the reference Gaussian state tau is not provided; assumptions about tau may be necessary.
- The method to compute the logarithm of density matrices and their traces is not detailed, which can be computationally challenging.
- The problem does not specify whether the coherent states |alpha> are normalized or any basis representation.
- Potential ambiguity in the phase phi's domain and its effect on normalization and state properties.
- The problem assumes familiarity with quantum states, density matrices, and relative entropy without elaboration on computational techniques
'''
    # task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, "Problem: Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][]. Entry_point: min_cost", task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, str(task_queue[0].content) + "\n\nDetailed Analysis: \n" + str(detailed_analysis), task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    # print(task_queue_tmp)
    output_description = get_global("global_output_description")
    max_attempt = 2
    acc_oracle_verifier_list = [0]
    total_time = 0
    final_results_path = f'results/{args.dataset}/recheck_mas/{meta_model}_{global_node_model}/final_results_{example_id}.json'
    result_path = Path(final_results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = []
    max_score = 0
    next_solution = {
        'code': ""
    }
    
    next_solution = {
        'code': '''
async def forward_166(self, taskInfo):
    logs = []
    loop_results = {"stage_0": {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": []}}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Construct the density matrix rho of the Schrödinger cat state |psi> for given phi and alpha, "
            "including calculation of normalization constant N. Use phi = -pi/4 and alpha = 0.5 from taskInfo. "
            "Provide detailed step-by-step reasoning and final expression for rho."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.5,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id="stage_0.subtask_0.iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        loop_results["stage_0"]["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Construct the reference Gaussian state density matrix tau corresponding to the Schrödinger cat state parameters. "
            "Use the output from Sub-task 0 and taskInfo to guide the construction. Provide detailed reasoning and final form of tau."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"]],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1.iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        loop_results["stage_0"]["subtask_1"].append(results_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Compute the relative entropy measure del_b = trace(rho ln rho) - trace(tau ln tau) "
            "using the constructed density matrices rho and tau from previous subtasks. Provide detailed calculation steps and numerical result."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"], results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2.iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        loop_results["stage_0"]["subtask_2"].append(results_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Refine and simplify the computed relative entropy result from Sub-task 2 to produce a clear intermediate numerical value for non-Gaussianity (nG). "
            "Provide concise final numerical value with reasoning."
        )
        cot_agent_desc_0_3 = {
            "instruction": cot_instruction_0_3,
            "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id="stage_0.subtask_3.iter_{}".format(iteration),
            cot_agent_desc=cot_agent_desc_0_3
        )
        logs.append(log_0_3)
        loop_results["stage_0"]["subtask_3"].append(results_0_3)

    aggregate_instruction_1_0 = (
        "Sub-task 0: Evaluate the collection of candidate nG values from stage_0 subtasks and select the best candidate that satisfies accuracy and consistency criteria. "
        "Use all refined numerical values from stage_0.subtask_3 iterations."
    )
    aggregate_desc_1_0 = {
        "instruction": aggregate_instruction_1_0,
        "input": [taskInfo] + [res["thinking"] for res in loop_results["stage_0"]["subtask_3"]],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from stage_0.subtask_3"]
    }
    results_1_0, log_1_0 = await self.aggregate(
        subtask_id="stage_1.subtask_0",
        aggregate_desc=aggregate_desc_1_0
    )
    logs.append(log_1_0)

    final_answer = await self.make_final_answer(results_1_0["thinking"], results_1_0["answer"])
    return final_answer, logs
        '''
    }
    
    print(f'============Initial Example: {example_id}=================')
    default_global_n = get_global("global_n")
    default_global_n[str(example_id)] = f"Test_multiple_times_{example_id}"
    set_global("global_n", default_global_n)

    global_n = get_global("global_n")
    global_n = global_n[str(example_id)]
    global_ns.append(global_n)
    
    # TODO: Load corresponding mas

    try:
        next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
        acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"])
        total_execution_time += total_time
    except Exception as e:
        print("Error: ", str(e))
        error_trace = traceback.format_exc()
        print("Full error trace:\n", error_trace)
        return -1, -1, -1, ""
    
    # judge_path = os.path.join(args.save_dir, f"{expr_name}_Test_multiple_time_{example_id}_full_response")
    # with open(judge_path, 'w') as judge_file:
    #     judge_file.write(f'Question: {task_queue[0].content}\nIteration: Test_multiple_time\nFull Response:{raw_results}')

    if global_use_oracle_verifier:
        acc_list = acc_oracle_verifier_list
    else:
        acc_list = acc_model_verifier_list

    print(f"acc_list:", acc_list)
    print(f"mean acc_list:", np.mean(acc_list))
    # print(f"bootstrap_confidence_interval: {fitness_str}")

    if 'swe_bench' in args.dataset:
        extracted_answer = final_reponse[0].split('\n\nAnswer:', 1)[-1].strip()
        if '<patch>' in extracted_answer:
            extracted_answer = extract_xml(extracted_answer, 'patch').strip()   
    else:
        extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

    if '[TOO_HARD]' in extracted_answer:
        extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        
    # memory.append({extracted_answer:fitness_str})
    # print(f'save json to {mem_path}')
    # with open(mem_path, 'w') as json_file:
    #     json.dump(memory, json_file, indent=4)
        
    print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))

    with open(oracle_acc_result_path, "a+") as fh:
        fh.write(f'experiemnt {example_id}: 1 (initial Test_multiple_times): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')
    
    max_score = max(max_score, acc_oracle_verifier_list[0])
            
    end_time_ = time.time()
    total_time = end_time_ - start_time_        
    
    final_results.append({
        "example_id": example_id,
        "score": max_score,
        "total_time": total_time,
        "total_execution_time": total_execution_time,
        "max_cost": get_global("global_COST_TOTAL"),
        "max_execution_cost": get_global("global_COST_EXECUTION")
    })
    
    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
            
    return acc_oracle_verifier_list[0], total_time, total_execution_time, result_path
        