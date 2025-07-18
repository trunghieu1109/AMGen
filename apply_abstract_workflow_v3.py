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
            
async def apply_abstract_workflow_enhance(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None, date_time=""):

    if example_id < 150:
        return 0, 0, 0, ""

    start_time_ = time.time()
    total_execution_time = 0
    
    global_node_model = get_global("global_node_model")
    save_file = "dev_26_refactored"
    abstract_mas_path = "workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2"
    
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
        'gpt-4.1-mini',
    ]

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
        for stage_id, stage in aw['flow'].items():
            subtask_names.add(stage['Title'])
        
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
    high_level_decomposition = await operator_templates.high_level_task_decomposition(meta_model, task_queue[0].content, 5)
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

    # # remove start and end step of control flow from mas
    
    # # filter sequential only mas
    # sequential_only_mas = []
    # loop_contain_mas = []
    # conditional_contain_mas = []
    
    # for idx, mas in enumerate(default_mas_chain):
    #     contain_loop = False
    #     contain_conditional = False
    #     # if 'start_conditional' in mas:
    #     #     conditional_contain_mas.append({
    #     #         'id': idx,
    #     #         'mas': mas
    #     #     })
    #     #     contain_conditional = True
        
    #     # if 'start_loop' in mas:
    #     #     loop_contain_mas.append({
    #     #         'id': idx,
    #     #         'mas': mas
    #     #     })
    #     #     contain_loop = True
        
    #     if not contain_conditional and not contain_loop:
    #         sequential_only_mas.append({
    #             'id': idx,
    #             'mas': mas
    #         })
            
    # async def choose_the_most_similar_mas(mas_chain, mas_chain_list, abstract_workflow_):
    #     if len(mas_chain_list) == 0:
    #         return None
        
    #     mas_chain_no_flow = []
    #     for mas in mas_chain_list:
    #         mas_no_flow = []
    #         for step in mas['mas']:
    #             if isinstance(step, str) and (step.startswith("start") or step.startswith("end")):
    #                 continue
    #             mas_no_flow.append(step)
                
    #         mas_chain_no_flow.append({
    #             'id': mas['id'],
    #             'mas': mas_no_flow
    #         })
        
    #     # choose the most similar abtracted workflow, based on levenshtein distance  
    #     sorted_chains = sorted(mas_chain_no_flow, key=lambda mas: levenshtein_array_to_array(mas_chain, mas['mas']))
    #     # print(sorted_chains)
    #     closest_1 = sorted_chains[:1]

    #     workflow_index = 0
    #     max_levenshtein_distance = 0
    #     for idx, awd in enumerate(abstract_workflow_):
    #         if levenshtein_array_to_array(awd['chain'], closest_1[0]['mas']) > max_levenshtein_distance:
    #             max_levenshtein_distance = levenshtein_array_to_array(awd['chain'], closest_1[0]['mas'])
    #             workflow_index = closest_1[0]['id']
                
    #     workflow_index = [workflow_index]
        
    #     print("workflow index: ", workflow_index)
    #     filterd_workflow = [abstract_workflow_[idx] for idx in workflow_index]
    #     print("Filtered workflow: ", [fw['name'] for fw in filterd_workflow])
        
    #     return filterd_workflow

    # # choose the most similar sequential only mas
    # filtered_sequential_only_workflow = await choose_the_most_similar_mas(mas_chain, sequential_only_mas, abstract_workflow)
    # print("Query-based chain: ", mas_chain)
    # print("Similar chain: ", filtered_sequential_only_workflow[0]['chain'])
    # print("Levenshtein distance: ", levenshtein_array_to_array(mas_chain, filtered_sequential_only_workflow[0]['chain']))
    # with open(log_path, "a+", encoding="utf-8") as f:
        
    #     phase = "Abstract MAS choosing"
    #     content = f"Query-based chain: {mas_chain}\nSimilar chain: {filtered_sequential_only_workflow[0]['chain']}\nLevenshtein distance: {levenshtein_array_to_array(mas_chain, filtered_sequential_only_workflow[0]['chain'])}"
        
    #     f.write(f"\n============== {phase} ================\n")
    #     f.write(str(content))
    # return 1, 1, 1, ""
    
    # alternatives
    # compare to available abstracted workflow via levenshtein distance
    distance_mas_pairs = [
        (levenshtein_array_to_array(mas_chain, mas), mas)
        for mas in default_mas_chain
    ]

    min_distance = min(distance_mas_pairs, key=lambda x: x[0])[0]

    min_distance_candidates = [
        mas for dist, mas in distance_mas_pairs if dist == min_distance
    ]

    sorted_chains = sorted(min_distance_candidates, key=lambda mas: -len(mas))

    # Lấy 2 chuỗi giống nhất
    closest_2 = sorted_chains[:2]
    print("MAS Chain: ", closest_2)
    print("Origin mas chain: ", mas_chain)
    for idx, cls_ in enumerate(closest_2):
        
        print(f"Levenshtein distance {idx}: ", levenshtein_array_to_array(mas_chain, cls_))
    
    # get 2 the most similar workflow
    workflow_index = []
    for idx, awd in enumerate(abstract_workflow):
        if str(awd['chain']) == str(closest_2[0]):
            workflow_index.append(idx)
            
        # if str(awd['chain']) == str(closest_2[1]):
        #     workflow_index.append(idx)

    if len(workflow_index) == 0:
        workflow_index = [random.randint(0, len(default_mas_chain) - 1)]
    
    workflow_index = [10]

    print("workflow index: ", workflow_index)
    # return 1, 1, 1, ""
    
    filterd_workflow = [abstract_workflow[idx] for idx in workflow_index]
    
    print("Filtered workflow: ", [fw['name'] for fw in filterd_workflow])
    # return 1, 1, 1, ""
    
    # # choose the most similar sequential only mas
    # filtered_loop_contain_workflow = await choose_the_most_similar_mas(mas_chain, loop_contain_mas, abstract_workflow)
    
    # # choose the most similar sequential only mas
    # filtered_conditional_contain_workflow = await choose_the_most_similar_mas(mas_chain, conditional_contain_mas, abstract_workflow)
    
    # print("\n========= Filtered Sequential Workflow =========\n", filtered_sequential_only_workflow[0]['flow'])
    # print("\n========= Filtered Loop Workflow =========\n", filtered_loop_contain_workflow[0]['flow'])
    
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
    if filterd_workflow:
        with open(filterd_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
            sequential_only_aw_ = json.load(f)
        sequential_only_aw = {
            'flow': filterd_workflow[0]['flow'],
            'specific_workflow': sequential_only_aw_
        }
            
    # loop_contain_aw = None
    # if filtered_loop_contain_workflow:
    #     with open(filtered_loop_contain_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
    #         loop_contain_aw_ = json.load(f)
        
    #     loop_contain_aw = {
    #        'flow': filtered_loop_contain_workflow[0]['flow'],
    #        'specific_workflow': loop_contain_aw_
    #     }
        
    # conditional_contain_aw = None
    # if filtered_conditional_contain_workflow:
    #     with open(filtered_conditional_contain_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
    #         conditional_contain_aw_ = json.load(f)

    #     conditional_contain_aw = {
    #         'flow': filtered_conditional_contain_workflow[0]['flow'],
    #         'specific_workflow': conditional_contain_aw_
    #     }
            
    print(sequential_only_aw)
    # print(loop_contain_aw)
    # print(conditional_contain_aw)
    
    # merge_filtered_aw = await merge_filtered_workflow(meta_model, sequential_only_aw, loop_contain_aw, conditional_contain_aw)
    dependencies_and_agent_collaboration = []
    for idx, st in enumerate(sequential_only_aw['specific_workflow']):
        dependencies_and_agent_collaboration.append({
            'stage_id': f'stage_{idx}',
            'agent_collaboration': st['agent_collaboration'],
            'dependencies': [f"From stage_{idx} to {dep.replace("subtask", "stage")}" for dep in st['dependencies']]
        })
    
    print(dependencies_and_agent_collaboration)
    
    print(task_queue_tmp[0].content)
    print(dependencies_and_agent_collaboration)
    print(sequential_only_aw['flow'])
    
    # decompose query into multiple subtasks
    task_decomposition = await operator_templates.specific_task_decomposition(meta_model, task_queue_tmp[0].content, dependencies_and_agent_collaboration, aw_flow=sequential_only_aw['flow'])
    print("\n============= Task Decomposition: =============\n", task_decomposition)
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "Task Decomposition"
        content = task_decomposition
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    # return 1, 1, 1, ""
    stage_desc = str(sequential_only_aw['flow']).replace("subtask", "stage")
    
    # generate new workflow that concretized for this query
    next_solution = await operator_templates.generate_concretized_workflow(meta_model, task_decomposition, interaction_pattern, task_queue_tmp[0].content)
    print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
    
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
    # judge_path = os.path.join(args.save_dir, f"{expr_name}_Test New Architecture_{example_id}_full_response")
    # with open(judge_path, 'w') as judge_file:
    #     judge_file.write(f'Question: {task_queue[0].content}\nIteration: Test New Architecture\nFull Response:{raw_results}')

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
    # final_results.append({
    #     "example_id": example_id,
    #     "score": max_score,
    #     "max_cost": get_global("global_COST_TOTAL")
    # })
    
    '''
    ================================================== REFINE WORKFLOW ======================================================
    '''
    
    # if the first concretized workflow is not optimal, try to evaluate and refine it
    if int(max_score) == 0:
        subtask_desc = logs
            
        # get evaluation from many llm experts
        evaluation = await operator_templates.evaluate_workflow(task_queue_tmp[0].content, subtask_desc, current_ans, output_description, verifier_hub)
        with open(log_path, "a+", encoding="utf-8") as f:
        
            phase = "Evaluation from verifiers"
            content = evaluation
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))

        refined_task_decomposition = await operator_templates.specific_task_decomposition(meta_model, task_queue_tmp[0].content, dependencies_and_agent_collaboration, evaluation=evaluation, prev_task_decomposition=task_decomposition, is_refinement=True)
        
        print("\n============= Refined Task Decomposition: =============\n", refined_task_decomposition)
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
        
# async def test_operator(args, expr_name, example_id, task_queue, meta_model, verifier_model, pattern = None):

#     questions = get_global("global_questions")
#     questions = questions[str(example_id)]
#     global_node_model = get_global("global_node_model")
    
#     cost_per_query = get_global("global_COST_TOTAL_per_query")
#     cost_per_query[str(example_id)] = 0.0
#     set_global("global_COST_TOTAL_per_query", cost_per_query)

#     print(f"problem length: {len(questions)}")
#     max_workers = min(len(questions), args.max_workers) if args.multiprocessing else 1

#     if args.dataset == 'gpqa_diamond':
#         task_queue = [Info(field_name, author, {"question": content.question, "choice1": content.choice1, "choice2": content.choice2, "choice3": content.choice3, "choice4": content.choice4}, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]
#     else:
#         task_queue = [Info(field_name, author, content, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]

#     set_global("global_max_workers", max_workers)
#     result_path = expr_name + f"{args.dataset}/{pattern}"
#     expr_name = expr_name + f"{args.dataset}/{pattern}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"

#     global_task_queue = get_global('global_task_queue')
#     global_task_queue[str(example_id)] = task_queue    
#     set_global("global_task_queue", global_task_queue)

#     next_solution_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_next_solution.json")
#     msg_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_msg.json")
#     mem_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_mem.json")
#     file_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_archive.json")
#     result_path = f'results/{args.dataset}/single_agent_baselines_v3/{pattern}/{meta_model}_{global_node_model}_{verifier_model}.results'
#     oracle_acc_result_path = f'results/{args.dataset}/single_agent_baselines_v3/{pattern}/{meta_model}_{global_node_model}_oracle.results'
#     oracle_acc_path = Path(oracle_acc_result_path)
#     oracle_acc_path.parent.mkdir(parents=True, exist_ok=True)
    
#     result_acc_path = Path(result_path)
#     result_acc_path.parent.mkdir(parents=True, exist_ok=True)

#     judge_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_judge")
#     reponse_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_reponse")
#     os.makedirs(os.path.dirname(judge_path), exist_ok=True)

#     print('file_path: ',file_path)
#     print('msg_path: ',msg_path)
#     print('result_path: ',result_path)
#     print('next_solution_path: ',next_solution_path)
#     print('oracle_acc_result_path: ',oracle_acc_result_path)
#     print('judge_path: ',judge_path)
#     print('reponse_path: ',reponse_path)
#     print('mem_path: ',mem_path)
    
#     global_judge_path = get_global('global_judge_path')
#     global_judge_path[str(example_id)] = judge_path

#     set_global("global_judge_path", global_judge_path)
    
#     global_reponse_path = get_global('global_reponse_path')
#     global_reponse_path[str(example_id)] = reponse_path

#     set_global("global_reponse_path", global_reponse_path)

#     if os.path.exists(mem_path):
#         with open(mem_path, 'r') as json_file:
#             memory = json.load(json_file)
#     else:
#         memory = []

#     if os.path.exists(reponse_path):
#         with open(reponse_path, 'r') as json_file:
#             global_response = json.load(json_file)
            
#         global_response_dict = get_global('global_response_dict')
#         global_response_dict[str(example_id)] = global_response
        
#         set_global("global_response_dict", global_response_dict)

#     global_use_oracle_verifier = get_global("global_use_oracle_verifier")

#     global_ns = []
    
#     final_results_path = f'results/{args.dataset}/single_agent_baselines_v3/{pattern}/{meta_model}_{global_node_model}/final_results_{example_id}.json'
#     result_path = Path(final_results_path)
#     result_path.parent.mkdir(parents=True, exist_ok=True)
#     final_results = []

#     judge_path = os.path.join(args.save_dir, f"{expr_name}_{pattern}_{args.option}_judge")
#     reponse_path = os.path.join(args.save_dir, f"{expr_name}_{pattern}_{args.option}_reponse")
#     print(f"================== Test single agent baselines {pattern} ===================")
#     default_global_n = get_global("global_n")
#     default_global_n[str(example_id)] = f"Baseline {pattern}"
#     set_global("global_n", default_global_n)

#     global_n = get_global("global_n")
#     global_n = global_n[str(example_id)]
#     global_ns.append(global_n)
    
#     blocks = get_init_archive(['cot', 'sc_cot', 'reflexion', 'debate'])
#     workflow = blocks[pattern]
#     try:
#         workflow["code"] = '''
# async def forward(self, taskInfo):
#     print("Task Requirement: ", taskInfo)
#     # Initialize lists to keep track of sub-tasks and agents
#     sub_tasks = []
#     agents = []
#     logs =  []
    
#     programmer_instruction1 = "Sub-task 1: Generate Python runnable code that addresses the following problem: [problem1]"
#     programmer_desc1 = {
#         'instruction': programmer_instruction1, 
#         'input': [taskInfo], 
#         'temperature': 0.0, 
#         'context': ["user query"],
#         'entry_point': "solve"
#     }
#     results1 = await self.programmer(
#         subtask_id="subtask_1", 
#         programmer_desc=programmer_desc1
#     )
    
#     agents.append(f"Programmer Agent {results1['programmer_agent'].id}, generate code for problem [problem #1], thinking: {results1['thinking'].content}; answer: {results1['answer'].content}, executing reults: {results1['exec_result']}")
#     sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}; output - {results1['exec_result']}")
#     logs.append(results1['subtask_desc'])
    
#     final_answer = await self.make_final_answer(results1['thinking'], results1['answer'], sub_tasks, agents)
#     return final_answer, logs
# '''
#         acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, _, _, _, total_time = await evaluate_forward_fn(args, example_id, workflow["code"])
#     except Exception as e:
#         print("Error: ", str(e))
#         error_trace = traceback.format_exc()
#         print("Full error trace:\n", error_trace)
    
#     return 0, 0, ""

async def recheck_mas(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None):

    if example_id != 156:
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
- The problem context is an outbreak of a viral infectious disease caused by a retrovirus in a city.
- The goal is to design a molecular diagnostic kit for quick detection.
- Four choices are provided, each describing a different approach:
  * Choice 1: Identify virus by DNA sequencing, then develop a PCR kit.
  * Choice 2: Identify IgG antibodies, then develop an ELISA kit targeting IgG antibodies.
  * Choice 3: Identify virus using symptom information from patients, then design a nested PCR kit.
  * Choice 4: Identify virus by cDNA sequencing, then develop a real-time PCR kit.
- Key entities: virus (retrovirus), DNA sequencing, cDNA sequencing, PCR, nested PCR, real-time PCR, IgG antibodies, ELISA.

2. Analyze Relationships Between Components:
- Identification methods vary: direct viral genetic material sequencing (DNA or cDNA), antibody detection, or symptom-based inference.
- Diagnostic methods depend on identification: PCR-based kits (standard, nested, real-time) or ELISA targeting antibodies.
- Constraints include speed and accuracy of diagnosis.
- The choice of sequencing (DNA vs cDNA) relates to the retrovirus lifecycle (RNA genome requiring reverse transcription).
- Antibody detection (IgG) reflects host immune response rather than direct viral detection.
- Symptom-based identification is indirect and may affect specificity.
- These components influence the diagnostic kit's design, sensitivity, specificity, and applicability.

3. Identify the Field of Study:
- Primary domain: Molecular biology and virology.
- Subfields: Diagnostic assay development, immunology (antibody detection), molecular genetics (sequencing, PCR techniques).
- Related fields: Biotechnology, clinical diagnostics, epidemiology.
- Applications: Infectious disease diagnosis, public health response, biomedical research.

4. Highlight Aspects Needing Clarification:
- The problem does not specify the viral genome type explicitly (though retrovirus implies RNA genome).
- The rationale for choosing DNA sequencing versus cDNA sequencing is not detailed.
- The reliability of symptom-based virus identification is ambiguous.
- The timing and stage of infection affecting antibody presence (IgG) is not mentioned.
- Potential challenges include differentiating between direct viral detection and immune response detection, and the technical feasibility of each method in the outbreak context.
    '''
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, task_queue[0].content, task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
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
async def forward(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Understand the nature of the retrovirus genome and implications for molecular diagnostics, "
        "including whether the viral genome is RNA or DNA and the need for reverse transcription."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent understanding of the retrovirus genome type and its implications."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)
    
    final_answer = await self.make_final_answer(results1['thinking'], results1['answer'])
    return final_answer, logs

    cot_sc_instruction2 = (
        "Sub-task 2: Review and summarize the principles and requirements of different diagnostic methods "
        "(DNA sequencing, cDNA sequencing, PCR variants, ELISA) in the context of retroviral detection, "
        "based on the viral genome nature from Sub-task 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent summary of diagnostic methods suitable for retroviral detection."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Integrate knowledge of viral genome type and diagnostic methods to design a molecular diagnostic workflow, "
        "including sample preparation, target identification, and assay development options."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of the proposed molecular diagnostic workflow design, "
        "considering the viral genome and diagnostic methods."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate and select the most appropriate diagnostic approach for quick and accurate detection of the retrovirus, "
        "considering assay sensitivity, specificity, speed, and feasibility based on the designed workflow from Sub-task 3."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best diagnostic approach for the retrovirus outbreak scenario."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Formulate a final design recommendation for the molecular diagnostic kit, "
        "specifying the identification method and diagnostic assay type best suited for the retrovirus outbreak scenario, "
        "based on the evaluation from Sub-task 4."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final recommended molecular diagnostic kit design."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
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
        