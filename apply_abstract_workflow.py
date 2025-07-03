import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
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
import pandas as pd
import common
from common import HTML_JINJA, get_init_archive, get_prompt, get_reflexion_prompt, SingleEvalResult, get_reflexion_after_eval
from common import get_json_response_from_gpt, get_json_response_from_gpt_reflect, _pack_message
from utils import random_id, bootstrap_confidence_interval
from common import ANSWER_PATTERN, shorten_context, merge_context
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
from abstract_v2 import MASAbstraction, levenshtein_array_to_array
import asyncio

client = openai.OpenAI()

Info = namedtuple('Info', ['name', 'author', 'content', 'prompt', 'sub_tasks', 'agents', 'iteration_idx'])

ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

ABSTRACTED_WORKFLOW_TEMPLATE = '''
async def forward(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []
    logs =  []
    
    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 1: <Fill the stage 1's stage_name>]
    [Objective] 
    - <Describe in detail the abstracted objective of stage 1.>
    - <Describe in detail the abstracted objective of stage 1.>
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage 1.>
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    # Sub-task 1: Analyze first expression/data component with self-consistency
    cot_instruction = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {{
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, analyzing [expression #1], thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    subtask_desc1['response'] = {{
        "thinking": thinking1,
        "answer": answer1
    }}
    logs.append(subtask_desc1)
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 2: <Fill the stage 2's stage_name>]
    [Objective] 
    - <Describe in detail the abstracted objective of stage 2.>
    - <Describe in detail the abstracted objective of stage 2.>
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage 2.>
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 3: <Fill the stage 3's stage_name>]
    [Objective] 
    - <Describe in detail the abstracted objective of stage 3.>
    - <Describe in detail the abstracted objective of stage 3.>
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage 3.>
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>

    <Continue with next stages>
    
    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage n: <Fill the stage n's stage_name>]
    [Objective] 
    - <Describe in detail the abstracted objective of stage n.>
    - <Describe in detail the abstracted objective of stage n.>
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage n.>
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    final_answer = await self.make_final_answer(thinkingn, answern, sub_tasks, agents)
    return final_answer, logs
'''

class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model=None, temperature=None) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature
        # give each instance a unique id
        self.id = random_id()
        
    async def extract_pattern(self, prompt):
        # pattern = r"\s*(.*?)\s*\n\nRelated original question"
        pattern = r"Given the above, answer the following question: \s*(.*?)\s*\n\n"

        sub_question = prompt[-1]['content']
        match = re.search(pattern, sub_question, re.DOTALL)
        extracted_question = match.group(1)

        return extracted_question

    async def generate_prompt(self, input_infos, instruction, is_sub_task=False) -> str:

        global_node_model = get_global("global_node_model")
        global_output_description = get_global("global_output_description")
        global_FORMAT_INST = get_global("global_FORMAT_INST")

        global_format_choice = get_global("global_format_choice")

        if global_format_choice == 'json':
            output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. {global_output_description}" for key in self.output_fields}
        elif global_format_choice == 'xml':
            output_fields_and_description = '\n'.join([f"<{key}> [Your {key}.] </{key}>" if not 'answer' in key else f"<{key}> [Your {key}. {global_output_description}] </{key}>\n" for key in self.output_fields])
        else:
            raise NotImplementedError

        system_prompt = ROLE_DESC(self.role) + "\n\n" + global_FORMAT_INST(output_fields_and_description)
        

        # construct input infos text
        input_infos_text = ''
        prev_extracted_question = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, prompt, _, _, iteration_idx) = input_info
            else:
                continue
            if author == await self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                if is_sub_task: 
                    input_infos_text += f'Related original question:\n\n{content}. \n\nRelated sub-task questions and answers:\n\n'
                else:
                    input_infos_text += f'{content}\n\n'
            elif iteration_idx != -1:
                if is_sub_task and prompt is not None: 
                    extracted_question =  await self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question
                    else:
                        input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'

                else:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                if is_sub_task and prompt is not None: 
                    extracted_question = await self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question # we do not want to duplicate the prompt
                    else:
                        input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        if is_sub_task: 


            if global_format_choice == 'json':

                prompt = input_infos_text + f'Given the above, answer the following question: {instruction}\n\n Return your answer in the "answer" entry and justify detailedly why you think so in the "thinking" entry. Answer is a string include the answer for this query'# instruction (sub-task in above)
                # prompt = input_infos_text + f'''Given the above, answer the following question: {instruction} \n\n then justify completely and detailedly, step-by-step why you think so in the "thinking" entry. 
                # If stuck in loop and cannot to solve this problem, stop and add [TOO HARD] in the last position.
                # Again, your task is only to answer the question {instruction} and explaination.'''# instruction (sub-task in above)

            elif global_format_choice == 'xml':

                prompt = input_infos_text + f"""Given the above, answer the following question: {instruction}\n\n 
                
                If the question is too complicated or informaion is missing, you still need to give your best guess but add (1) an additional mark [TOO_HARD] in the next line of your final answer (2) information request or decomposison suggestion in the next line of the [TOO_HARD] mark, in the "answer" entry. In the "thinking", justify why you think so. Following the format below:
                
                "answer" entry: [Your best guess, e.g., 300]\n[TOO_HARD]\nSuggestion: [your suggestion]
                "thinking" entry:  [why you thinking is is too complicated or missing information. How to you arrive your best guess regardless]

                Otherwise, give your answer and thinking normally.

                "answer" entry: [your answer]
                "thinking" entry: [How do you arrive your answer]

                IMPORTANT: You need to give your best guess in both cases. Do not give [TOO_HARD] directly but always give your best guess first

                """
            else:
                raise NotImplementedError


        else:
            prompt = input_infos_text + instruction
        return system_prompt, prompt

    async def query(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False) -> dict:
        

        system_prompt, prompt = await self.generate_prompt(input_infos, instruction, is_sub_task=is_sub_task)

        prompt = [
            await _pack_message(content=system_prompt, role="system"),
            await _pack_message(content=prompt, role="user")]
        # use system prompt

        response_json ,_ = await get_json_response_from_gpt(prompt, self.model, self.output_fields, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, await self.__repr__(), value, prompt, None, None, iteration_idx)
            output_infos.append(info)
        return output_infos

    async def __repr__(self):
        return f"{self.agent_name} {self.id}"

    async def __call__(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False):
        return await self.query(input_infos, instruction, iteration_idx=iteration_idx,  is_sub_task=is_sub_task)




class AgentSystem():
    def __init__(self) -> None:
        pass

    async def make_final_answer(self, thinking, answer, sub_tasks=None, agents=None):

        name = thinking.name
        author = thinking.author
        prompt = thinking.prompt
        iteration_idx = thinking.iteration_idx

        if type(answer) == str:
            answer_content = answer
        else:
            answer_content = answer.content

        if agents is None: # this means sub_task is None, according to the propose prompt
            sub_tasks, agents = agents, sub_tasks

        if sub_tasks is None and agents is None:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, None, None, iteration_idx)
        else:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, '\n<SEPERATOR>\n'.join(sub_tasks), '\n<SEPERATOR>\n'.join(agents), iteration_idx)
        return final_answer

async def evaluate_forward_fn(args, example_id, forward_str):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py


    print('forward_str: ',forward_str)

    # if you want debug, remove the section so that you can see the detailed error line
    namespace = {}
    try:
        safe_forward_str = forward_str.replace("\\", "\\\\")
        exec(safe_forward_str, globals(), namespace)
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
    
    global_reponse_path = get_global("global_reponse_path")
    global_reponse_path = global_reponse_path[str(example_id)]
    
    global_response_dict = get_global("global_response_dict")
    global_response_dict = global_response_dict[str(example_id)]
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
            global_reponse_path,
            global_response_dict,
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
        
async def test_mas_zero_workflow(args, expr_name, example_id, task_queue, meta_model, verifier_model, mas_zero_workflow = None):

    questions = get_global("global_questions")
    questions = questions[str(example_id)]
    global_node_model = get_global("global_node_model")

    print(f"problem length: {len(questions)}")
    max_workers = min(len(questions), args.max_workers) if args.multiprocessing else 1

    task_queue = [Info(field_name, author, content, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]

    set_global("global_max_workers", max_workers)
    
    expr_name = expr_name + f"{args.dataset}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"
    
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue    
    set_global("global_task_queue", global_task_queue)

    next_solution_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_next_solution.json")
    msg_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_msg.json")
    mem_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_mem.json")
    file_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_archive.json")
    result_path = f'results/{args.dataset}/mas_zero/{meta_model}_{global_node_model}_{verifier_model}.results'
    oracle_acc_result_path = f'results/{args.dataset}/mas_zero/{meta_model}_{global_node_model}_oracle.results'
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
    for _id, workflow in enumerate(mas_zero_workflow):
        workflow_id = int(_id / 5)
        n = int(_id % 5)
        
        if workflow_id != example_id:
            continue
        
        if n != 0:
            continue
        
#         user_prompt_generate_workflow = f"""
# You are a meticulous static code analyzer specialized in detecting the use of undeclared variables in Python functions.
# Your task is to examine the provided Python function and ensure that:
# 1. Every variable is declared before it is used, including:
# - Local variables
# - Function parameters
# - Arguments passed to internal function calls
# - Loop variables, list comprehension variables, etc.
# 2. If any variable is used before being declared, or is never declared within the scope of the function, you must provide a detailed report that includes:
# - The name of the variable
# - The line where the issue occurs
# - A precise explanation of why the variable is considered undeclared
# 3. Also, watch out for:
# - Typos or incorrect references to variable names
# - Function calls using undeclared variables as arguments

# Example (for testing purposes):

# ```python
# def example(a, b):
#     total = a + b
    
#     total += c  # 'c' is undeclared
    
#     for i in range(n):  # 'n' is undeclared
#         sum += i
        
#     return total + sum  # 'sum' is undeclared
# ```

# Note: This function is injected into a class, so `self....` is declared and it not an error.

# Function:
# {workflow['code']}

# Reply EXACTLY with the following JSON format. Return `thought`, which is your thought and `is_error`, which is `True` or `False`.
# {{
#     "thought": "Your thought.", 
#     "is_error": True | False
# }}

# DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!
#         """
        
#         msg_list = [
#             # {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt_generate_workflow},
#         ]

#         checking = get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'is_error'], 0.0)
        
#         print(f"Workflow: {workflow_id}, iteration: {n}, is_error = {checking['is_error']} \n thought: {checking['thought']}")
        
#         if str(checking['is_error']) == 'True':
#             judge_path = os.path.join(args.save_dir, f"{expr_name}_{workflow_id}_iteration{n}_full_response")
#             with open(judge_path, 'w') as judge_file:
#                 judge_file.write(f'Question: {task_queue[0].content}\nIteration: {n}\nVariables Checking: {checking['thought']}')
#             continue
        
        judge_path = os.path.join(args.save_dir, f"{expr_name}_{workflow_id}_iteration{n}_{args.option}_judge")
        reponse_path = os.path.join(args.save_dir, f"{expr_name}_{workflow_id}_iteration{n}_{args.option}_reponse")

        print(f"================== Test workflow for example {workflow_id}, iteration: {n} ===================")
        default_global_n = get_global("global_n")
        default_global_n[str(example_id)] = f"MAS-ZERO Workflow {workflow_id}_Iteration_{n}"
        set_global("global_n", default_global_n)

        global_n = get_global("global_n")
        global_n = global_n[str(example_id)]
        global_ns.append(global_n)

        try:
            workflow["code"] = workflow['code'].replace("forward", f"forward_{example_id}")
            acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results = await evaluate_forward_fn(args, example_id, workflow["code"])
        except Exception as e:
            print("Error: ", str(e))
            continue
        
        judge_path = os.path.join(args.save_dir, f"{expr_name}_{workflow_id}_iteration{n}_full_response")
        with open(judge_path, 'w') as judge_file:
            judge_file.write(f'Question: {task_queue[0].content}\nIteration: {n}\nFull Response:{raw_results}')
            
        with open(oracle_acc_result_path, "a+") as fh:
            fh.write(f'experiemnt {example_id}: 1 (initial MAS-ZERO_Workflow_{workflow_id}_iteration_{n}): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')

        #TODO: can we somehow also log acc_oracle_verifier_list so that we can know how accurate acc_model_verifier_list is?
        if global_use_oracle_verifier:
            acc_list = acc_oracle_verifier_list
        else:
            acc_list = acc_model_verifier_list


        if args.defer_verifier:
            fitness_str = bootstrap_confidence_interval([0.0])
            workflow["acc"] = np.mean([0.0])

        else:
            fitness_str = bootstrap_confidence_interval(acc_list)
            workflow["acc"] = np.mean(acc_list)

        workflow["fitness"] = fitness_str
        workflow["total_cost"] = get_global("global_COST_TOTAL")

        print(f"acc_list:", acc_list)
        print(f"mean acc_list:", np.mean(acc_list))
        print(f"bootstrap_confidence_interval: {fitness_str}")


        if 'swe_bench' in args.dataset:
            extracted_answer = final_reponse[0].split('\n\nAnswer:', 1)[-1].strip()
            if '<patch>' in extracted_answer:
                extracted_answer = extract_xml(extracted_answer, 'patch').strip()   
        else:
            extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

        if '[TOO_HARD]' in extracted_answer: # we cannot add [TOO_HARD] in memory
            extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        
        memory.append({extracted_answer:fitness_str})
        print(f'save json to {mem_path}')
        with open(mem_path, 'w') as json_file:
            json.dump(memory, json_file, indent=4)

        # save results

        report_filename = os.path.join(args.save_dir, f'{expr_name}_MAS-ZERO_Workflow_{workflow_id}_{n}_{args.option}_debug.html')
        print(f"Writing report to {report_filename}")
        with open(report_filename, "w") as fh:
            fh.write(common.make_report(results))
        metrics = results.metrics | {"score": results.score}
        print('metrics: ',metrics)
        print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))
            
async def apply_abstract_workflow_enhance(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None):

    # if example_id < 150:
    #     return 0, 0, ""
    start_time_ = time.time()
    questions = get_global("global_questions")
    questions = questions[str(example_id)]
    global_node_model = get_global("global_node_model")

    print(f"problem length: {len(questions)}")
    max_workers = min(len(questions), args.max_workers) if args.multiprocessing else 1
    
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-aime24/abstracted_workflow/workflow_chains.json', 'r', encoding='utf-8') as f:
        default_mas_chain = json.load(f)
    result_path = expr_name + f"{args.dataset}"
    expr_name = expr_name + f"{args.dataset}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"

    if args.dataset == 'gpqa_diamond':
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
    oracle_acc_result_path = f'results/{args.dataset}/abstract_workflow_refined/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_result_path = f'results/{args.dataset}/abstract_workflow_refined/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_path = Path(oracle_acc_result_path)
    oracle_acc_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_acc_path = Path(result_path)
    result_acc_path.parent.mkdir(parents=True, exist_ok=True)

    judge_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_judge")
    reponse_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_reponse")
    os.makedirs(os.path.dirname(judge_path), exist_ok=True)
    
    interaction_pattern = """
Sample agent iteraction pattern:
Chain-of-Thought: 
```python
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {{
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, consider/calculate all possible scenarios of [problem #1], thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    subtask_desc1['response'] = {{
        "thinking": thinking1,
        "answer": answer1
    }}
    logs.append(subtask_desc1)
```

Self-Consistency Chain-of-Thought:
```python
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {{}}
    answermapping = {{}}
    subtask_desc2 = {{
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }}
    for i in range(N):
        # Each CoT-SC agent tries to calculate all possible cases independently
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}}, consider all possible cases of [problem #2], thinking: {{thinking2.content}}; answer: {{answer2.content}}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
        
    # The most common answer is chosen for consistency and accuracy.
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}}; answer - {{answer2.content}}")
    subtask_desc2['response'] = {{
        "thinking": thinking2,
        "answer": answer2
    }}
    logs.append(subtask_desc2)
```

Reflexion:
```python
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    subtask_desc3 = {{
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }}
    
    # Generate the first version
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {{cot_agent.id}}, filter valid scenarios of [problem], thinking: {{thinking3.content}}; answer: {{answer3.content}}")

    for i in range(N_max):
        # Critic agent debates and criticizes pros and cons of previous version
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "please review the [valid scenarios] filtering and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {{critic_agent.id}}, providing feedback, thinking: {{feedback.content}}; answer: {{correct.content}}")
        if correct.content == "True":
            break
        
        # Include previous version and feedback from critic agent as input
        cot_inputs.extend([thinking3, answer3, feedback])
        
        # Generate new version based on previous version and feedback
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {{cot_agent.id}}, refining valid scenarios of [problem], thinking: {{thinking3.content}}; answer: {{answer3.content}}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {{thinking3.content}}; answer - {{answer3.content}}")
    subtask_desc3['response'] = {{
        "thinking": thinking3,
        "answer": answer3
    }}
    logs.append(subtask_desc3)
```

Debate:
```python
    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    subtask_desc5 = {{
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4",
        "agent_collaboration": "Debate"
    }}
    
    for r in range(N_max_5):
        # N_max_5 rounds of debating
        for i, agent in enumerate(debate_agents_5):
            # Each agent proposes its solution
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                # Generate next solution based on comments and counter-arguments from other debaters
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {{agent.id}}, round {{r}}, converting [intermediate output] and calculating [final output], thinking: {{thinking5.content}}; answer: {{answer_5.content}}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    # Final decision agent makes final decision
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on [final output].", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating [final output], thinking: {{thinking5.content}}; answer: {{answer5.content}}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {{thinking5.content}}; answer - {{answer5.content}}")
    subtask_desc5['response'] = {{
        "thinking": thinking5,
        "answer": answer5
    }}
    logs.append(subtask_desc5)
```
    """

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
    
    # decompose task to high-level subtasks
    task_high_level_decomposition = f"""
You are an expert LLM assistant trained to decompose user queries to only many core subtasks list. 
Your task is to decompose user queries to only many core subtasks list.

Maximum 4 subtasks.

Input Query:
{task_queue[0].content}

Return your result in valid JSON format with the following structure. Each elements in `subtask_list` contains `objective`:
{{
    "subtask_list": [
        {{
            "objective": "The main description of this subtask"
        }}
    ]
}}
    """
    
    msg_list = [
        {"role": "user", "content": task_high_level_decomposition},
    ]

    high_level_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['subtask_list'], 0.0)
    
    print("high_level_decomposition: ", high_level_decomposition['subtask_list'])
    
    high_level_subtasks = [subtask['objective'] for subtask in high_level_decomposition['subtask_list']]
    
    # embeddings, postprocess and clustering the embeddings
    abstractor = MASAbstraction()
    abstracted_subtasks = await abstractor.abstract_task_decomposition(task_queue[0].content, high_level_subtasks)
    merged_subtasks = [f"{a_subtask['subtask_name']}: {a_subtask['abstracted_objective']}" for a_subtask in abstracted_subtasks]
    print(merged_subtasks)
    
    embeddings = await abstractor.embedding_subtask(merged_subtasks)
    
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-aime24/kmeans.pkl', 'rb') as f:
        kmeans = pkl.load(f)
    
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-aime24/pca.pkl', 'rb') as f:
        pca = pkl.load(f)
        
    normalized_embeddings = normalize(embeddings, norm="l2")
    reduced_embeddings = pca.transform(normalized_embeddings)
    
    cluster_id = kmeans.predict(reduced_embeddings)
    mas_chain = []
    for idx in cluster_id:
        if str(idx) not in mas_chain:
            mas_chain.append(str(idx))
            
    # compare to available abstracted workflow via levenshtein distance
    sorted_chains = sorted(default_mas_chain, key=lambda mas: levenshtein_array_to_array(mas_chain, mas))

    # Lấy 2 chuỗi giống nhất
    closest_2 = sorted_chains[:1]
    
    # get 2 the most similar workflow
    workflow_index = []
    for idx, awd in enumerate(abstract_workflow):
        if str(awd['chain']) == str(closest_2[0]):
            workflow_index.append(idx)
            
        # if str(awd['chain']) == str(closest_2[1]):
        #     workflow_index.append(idx)

    if len(workflow_index) == 0:
        workflow_index = [random.randint(0, len(default_mas_chain))]

    print("workflow index: ", workflow_index)
    
    filterd_workflow = [abstract_workflow[idx] for idx in workflow_index]
    
    print("Filtered workflow: ", [fw['name'] for fw in filterd_workflow])
    task_content = task_queue[0].content
    
    # task_content = task_content.replace("'", "")
    # task_content = task_content.replace("{", "")
    # task_content = task_content.replace("}", "")
    
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, task_content, task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    output_description = get_global("global_output_description")
    max_attempt = 2
    acc_oracle_verifier_list = [0]
    total_time = 0
    final_results_path = f'results/{args.dataset}/abstract_workflow_refined/{meta_model}_{global_node_model}/final_results_{example_id}.json'
    result_path = Path(final_results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = []
    max_score = 0
    for aw_id, aw in enumerate(filterd_workflow):
        evaluation = []
        task_decomposition = None
        next_solution = {
            'code': ""
        }
        for attempt in range (0, max_attempt):
            print(f'============Initial Archive: {aw["name"]}=================')
            default_global_n = get_global("global_n")
            default_global_n[str(example_id)] = f"{aw["name"]}_{example_id}"
            set_global("global_n", default_global_n)

            global_n = get_global("global_n")
            global_n = global_n[str(example_id)]
            global_ns.append(global_n)
            
            # decompose query into multiple subtasks
            
            user_prompt = f"""
You are an agent specialized in task decomposition. Your task is to analyze the provided query thoroughly and decompose it into subtasks and their dependencies to address the query effectively. You must incorporate feedback from previous decomposition attempts (evaluation) to improve the quality of the decomposition, avoiding past mistakes and addressing any identified issues.
[Query]
{task_queue_tmp[0].content}

[Output format]
{output_description}

[Instruction]
1. Decompose the query into fine-grained, detailed subtasks. Ensure the subtasks are simple, specific, and aligned with the model's capabilities, avoiding overly complex steps.
2. Structure the subtasks in a multi-turn reasoning sequence (e.g., step 1 → step 2), ensuring logical progression.
3. For each subtask, include:
    3.1. Objective: Clearly state the purpose of the subtask, incorporating all relevant context or conditions from previous subtasks to enhance performance.
    3.2. Dependencies: Specify which subtasks must be completed before the current subtask, ensuring proper sequencing.
4. Analyze the evaluation from previous decomposition attempts to identify weaknesses, redundancies, or gaps. Use this feedback to refine the decomposition, ensuring:
    4.1. Subtasks address any issues highlighted in the evaluation (e.g., overly broad tasks, missing dependencies, or unclear objectives).
    4.2. The decomposition improves upon previous attempts by being more precise, efficient, or comprehensive.
5. Incorporate domain expert feedback:
    5.1. Carefully review all feedback and evaluation from previous attempts.
    [Evaluation]
    {evaluation}
    5.2. Leverage any suggestions or expert critique to address prior issues such as:
        5.2.1. Missing dependencies.
        5.2.2. Overly broad or abstract subtasks.
        5.2.3. Redundant steps.
        5.2.4. Gaps in logic or execution order.
6. Refine the current decomposition by ensuring:
    6.1. Improved granularity, clarity, and logical sequencing over previous attempts.
    6.2. Outputs from one subtask are well-formed and suitable as inputs to the next, enhancing interoperability between subtasks.
    6.3. If a reasoning flaw is identified by expert evaluation, guide the agents' instructions to avoid repeating the same mistake.
    6.4. Use feedbacks from experts to guide agents's instructions.
7. If the evaluation is empty, treat this as the first attempt and focus on creating a clear, logical, and detailed decomposition.
8. Follow the provided workflow description to structure the decomposition process.
9. Ensure the decomposition is tailored to the query’s intent and context, leveraging any available information from previous tasks or the query itself.
10. Consider output format of each subtask, so that they could pass their output to other agents effectively.

[Previous Task Decomposition]
{task_decomposition}

<Abstract Workflow Description>
{aw['flow']}
</Abstract Workflow Description>

Return in JSON format, contains 'task_decomposition'. 
{{
    'task_decomposition': {{
        'stage_1': {{
            'subtask_1': {{
                'objective': "",
                'dependencies': []
            }},
            'subtask_2': {{
                'objective': "",
                'dependencies': [subtask_1]
            }}
        }},
        'stage_2': {{
            'subtask_3': {{
                'objective': "",
                'dependencies': [...]
            }},
            'subtask_4': {{
                'objective': "",
                'dependencies': [...]
            }}
        }}
    }}
}}
        """
        
            msg_list = [
                {"role": "user", "content": user_prompt},
            ]

            task_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['task_decomposition'], 0.0)
            for k, v in task_decomposition['task_decomposition'].items():
                for ks, vs in v.items():
                    vs['objective'] = vs['objective'].replace("{", "")
                    vs['objective'] = vs['objective'].replace("}", "")
                    vs['objective'] = vs['objective'].replace("'", "")
            print("\n============= Task Decomposition: =============\n", task_decomposition['task_decomposition'])
            stage_desc = str(aw).replace("subtask", "stage")
            # generate new workflow that concretized for this query
            user_prompt_generate_workflow = f"""
You are tasked with instantiating a concrete version of an abstract multi-stage workflow to solve complex queries using agent-based reasoning techniques (e.g., Chain-of-Thought, Self-Consistency, Reflexion, Debate). The workflow must be customized for the provided query and incorporate feedback from previous attempts to improve contextual usage, reasoning processes, and overall effectiveness.

[Important Notes]

1. The Abstract Workflow is an example implementation. You may adapt the agent interaction patterns (e.g., CoT, Self-Consistency CoT, Debate, Reflexion) to best suit the query’s complexity and requirements.
2. DO NOT USE LATEX FORMAT in the generated workflow.
3. Ensure the final part includes the make_final_answer function.
4. The generated code must include sufficient subtasks as planned in the task decomposition.
5. Incorporate feedback from the evaluation to address issues related to contextual usage and reasoning processes.
6. A stage may include multiple steps if necessary to achieve the query’s objectives.

[Feedback from Previous Attempt]
{evaluation}

[Previous workflow]
{next_solution['code']} 
Apply the feedbacks and suggestions from domain experts to improve your workflow. 

[Instructions]
You must follow the requirements below when generating the concrete workflow:

1. Return Format
Reply EXACTLY in the following JSON format. The generated code MUST be in function format (i.e., forward function) without comments.
{{
"code": "Your code."
}}
Ensure the response is a well-formed JSON object and includes all required fields.

2. Workflow Decomposition
    2.1. Follow the provided task decomposition to structure the workflow. 
    [Task Decomposition] {task_decomposition['task_decomposition']}
    2.2. Ensure each subtask is implemented as a step in the workflow, with clear objectives and dependencies as specified in the decomposition.

3. Agent Collaboration Patterns
    3.1. Implement agent interaction structures such as Chain-of-Thought (CoT), Self-Consistency CoT (SC-CoT), Reflexion, or Debate, selecting the most appropriate pattern for each subtask based on its complexity and the query’s requirements.
    3.2. Retain variables like self.node_model, self.debate_role, self.max_sc, and self.max_round with their correct roles.
    3.3. Refer to the provided interaction pattern examples for implementation guidance. 
    [Interaction Pattern] {interaction_pattern}

4. Placeholder Replacement
    4.1. Replace placeholders (e.g., [final answer], [condition #1]) with specific information relevant to the subtask. For example, if the query involves calculating the speed of a nucleus, replace [final answer] with “the speed of the nucleus.”
    4.2. Ensure reasoning goals are fully grounded in the query’s context for clarity and relevance.

5. Logic Preservation
Maintain the structure and logic of the abstract workflow:
    5.1. Preserve the same number of stages.
    5.2. Ensure stage transitions align with the inference objectives.
    5.3. Synthesize the final answer from intermediate steps.
    5.4. Print the result of each subtask using print("Step x: ", sub_tasks[-1]).
    5.5. Include from collections import Counter in the code.
6. Remove the comments:
    6.1. Avoid using comments in the generated code to keep it clean and concise.

7. Logging:
Log each subtask in the logs list, including:
    7.1. `subtask_id`, `instruction`, `context`, and `agent_collaboration` before agent execution.
    7.2. `response` (including `thinking` and `answer`) after agent execution.
    7.3. Ensure you logs enough information.
[Your Task]
Generate a concrete agentic workflow in Python code format, based on the Abstract Workflow, tailored to the query and informed by the evaluation feedback.

[Workflow Description]
{stage_desc}

[Workflow Template]
{ABSTRACTED_WORKFLOW_TEMPLATE}

[Query]
{task_queue_tmp[0].content}
            """
            
            msg_list = [
                {"role": "user", "content": user_prompt_generate_workflow},
            ]

            next_solution ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['code'], 0.0)
            print("================ Generated Multi-agent system ================\n", next_solution['code'])
            next_solution['code'] = next_solution['code'].replace("{{", "{")
            next_solution['code'] = next_solution['code'].replace("}}", "}")
            print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
            # evaluate multi agent system
            try:
                next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
                acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"])
            except Exception as e:
                print("Error: ", str(e))
                error_trace = traceback.format_exc()
                print("Full error trace:\n", error_trace)
                continue
            
            # print("========== LOGS ===========\n", logs)
            
            subtask_desc = logs
            
            if not acc_oracle_verifier_list[0] and attempt < max_attempt - 1:
                # TODO refine this workflow
                # pass
                prev_context = []
                reasoning_refinement_prompt = f"""
You are a Verification Agent tasked with reviewing the reasoning process of previous agents to identify errors, evaluate the sufficiency of subtask contexts, assess agent interactions, and suggest improvements to the agentic workflow. Your goal is to ensure the accuracy and completeness of the responses to the user query.
Based on feedback from professors, the current answer of this workflow is wrong. So, you need try to find the cause of this failure and then suggest the potential methods to improve the performance.

# Input
You will receive the following:
1. User Query: The original query provided by the user.
[User query]
{task_queue_tmp[0].content}
2. Subtasks Description:
    2.1. Instruction: The instructions provided for each subtask.
    2.2. Context: The context or input data used for each subtask.
    2.3. Response: The reasoning process (thinking steps) and the final answer for each subtask.
    2.4. Current Answer: The final answer provided to the user query.
[Subtask Description]
{subtask_desc}

[Current Answer]
{current_ans}
3. Agent Interaction Details: Information on how agents collaborated to produce the response (e.g., sequential, parallel, or iterative interactions).
4. Output Format: the output format of current query.
[Output format]
{output_description}

# Tasks
Perform the following tasks:
1. Reasoning Verification:
    1.1. Review the reasoning process (thinking steps) of each subtask.
    1.2. Identify any logical errors, incorrect assumptions, or missteps in the reasoning process.
    1.3. Pinpoint the specific step or reasoning flaw that led to the incorrect final answer, if applicable.
2. Context Sufficiency Evaluation:
    2.1. Assess whether the context provided for each subtask is sufficient to complete the task accurately.
    2.2. Identify any missing information, ambiguous context, or irrelevant details that may have affected the subtask's outcome.
    2.3. Suggest additional context or data that could improve the subtask's execution.
3. Agent Interaction Analysis:
    3.1. Evaluate the interaction between agents (e.g., how outputs from one subtask feed into another).
    3.2. Determine if the collaboration pattern used (e.g., sequential, parallel) was effective or if it contributed to errors.
    3.3. If a subtask failed, assess whether the failure was due to poor agent collaboration or insufficient information sharing.
4. Collaboration Pattern Recommendation:
    4.1. If a subtask failed or produced suboptimal results, evaluate whether adopting a different agent collaboration pattern could improve reasoning. Consider the following patterns:
    4.2. Chain-of-Thought (CoT): Structured step-by-step reasoning.
    4.3. Self-Consistency Chain-of-Thought (SC CoT): Generating multiple reasoning paths and selecting the most consistent.
    4.4. Debate: Agents argue different perspectives to refine the solution.
    4.5. Reflexion: Agents reflect on their reasoning to identify and correct errors.
Recommend the most suitable pattern for each subtask, with a brief justification.
5. Output Validation:
    5.1. Verify whether the output of each subtask adheres to the specified output format.
    5.2. Check if the output is accurate, complete, and aligned with the subtask's instructions and the user query.
    5.3. Identify any discrepancies or errors in the output.
6. Workflow Improvement Suggestions: Based on your analysis, provide actionable recommendations to improve the agentic workflow. Consider the following approaches:
    6.1. Decompose Subtasks: Break down complex subtasks into smaller, more manageable tasks that can be assigned to other agents.
    6.2. Change Collaboration Patterns: Suggest alternative collaboration patterns (e.g., CoT, SC CoT, Debate, Reflexion) for specific subtasks to enhance reasoning.
    6.3. Rewrite Instructions: Propose clearer, more specific instructions for subtasks to reduce ambiguity and improve execution.
    6.4. Refine Context: Recommend ways to improve the context of subtasks, such as reconnecting outputs from one subtask to the input of another for better coherence.

7. Output Requirements:  
Return a JSON object with two fields: 'feedback' and 'suggestion'.  
- **Feedback**: Provide a detailed, step-by-step reviewing the reasoning process of previous agents to identify errors and evaluation about the sufficiency of subtask contexts, assess agent interactions, and suggest improvements to the agentic workflow.
- **Suggestion**: Offer actionable recommendations to improve the agentic workflow.

**JSON Format**:  
{{
    "feedback": "Detailed evaluation of the agent’s context usage, reasoning process, collaboration effectiveness, instruction adherence, logical soundness, and output format effectiveness.",
    "suggestion": "Specific, actionable recommendations to improve the agentic workflow"
}}

# Guidelines
1. Be thorough and specific in identifying errors and proposing solutions.
2. Use clear, concise language to describe issues and recommendations.
3. Ensure recommendations are practical and actionable.
4. If no errors are found in a subtask, explicitly state that it was executed correctly.
5. Prioritize suggestions that address the root cause of errors in the reasoning process.
                """
                
                msg_list = [
                    {"role": "user", "content": reasoning_refinement_prompt},
                ]

                verifier_hub = [
                    'o4-mini',
                    'gpt-4o-mini-2024-07-18'
                ]

                # mas_feedback ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['suggestion', 'feedback'], 0.0)
                # print("================ MAS Refinement Feedback ================\n", mas_feedback['feedback'])
                # print("================ MAS Refinement suggestion ================\n", mas_feedback['suggestion'])
                # evaluation.append(mas_feedback)
                
                tasks = [
                    (verifier_model, asyncio.create_task(
                        get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['suggestion', 'feedback'], 0.0)
                    ))
                    for verifier_model in verifier_hub
                ]
                
                # Run all tasks in parallel and gather results
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                # Process results in order of verifiers
                for (verifier_model, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        print(f"================ Error for {verifier_model} ================\n", str(result))
                        continue
                    mas_feedback, _ = result
                    evaluation.append({
                        "model_name": verifier_model,
                        "evaluation": mas_feedback
                    })
            
                synthesizer_prompt = f"""
You are an advanced agent specialized in synthesizing evaluations from multiple verifier models to produce a single, cohesive feedback and suggestion output. Your task is to analyze the feedback and suggestions provided by a set of verifiers, resolve any conflicts, and generate a unified feedback and suggestion that is clear, concise, and actionable. The input evaluations come from a list of verifier models (verifier_hub) and are stored in an evaluation list, where each entry contains feedback and suggestion fields.

[Input Context]
1. Evaluations: A list of evaluation results, where each evaluation is a dictionary containing:
- feedback: A string describing issues or observations about the query or task performance.
- suggestion: A string proposing improvements or next steps.
These feedback come from many verifiers:
{evaluation}

Query: The original query or task context: {task_queue_tmp[0].content}

[Instructions]
1. Analyze Evaluations:
    1.1. Review the feedback and suggestion from each verifier in the evaluation list.
    1.2. Identify common themes, agreements, or contradictions across the verifiers’ outputs.
    1.3. Assess the relevance and quality of each feedback and suggestion relative to the query context.
2. Resolve Conflicts:
    2.1. If verifiers provide conflicting feedback or suggestions, prioritize based on:
    2.2. Relevance to the query’s intent and context.
    2.3. Consistency with common themes across verifiers.
    2.4. The credibility or specificity of the verifier’s output (e.g., prefer detailed, actionable suggestions over vague ones).
    2.5. If no clear resolution is possible, include a balanced summary of conflicting points and propose a reasonable compromise.
3. Synthesize Output:
    3.1. Combine the feedback into a single, defailed narrative, avoiding redundancy.
    3.2. Combine the suggestions into a single, actionable recommendation that integrates the best ideas from all verifiers and addresses the query’s goals.    
    3.3. Ensure the output is clear, avoids jargon unless necessary, and is structured for easy understanding.
The Synthesized output must be detailed (failure in which subtasks?, why it was failed?, does the problem come from reasoning process?, does the problem come from agent collaboration patterns?)
Provide a bullet-point formatted list including all the feedbacks and suggestions from experts.
About the suggestions, for each of them, describe detailed how to modify.
4. Output Format:
Return the result in JSON format with the following structure:
{{
    "combined_feedback": "Issues or observations from all verifiers that are limitations of current workflow.",
    "combined_suggestion": "A single, actionable recommendation integrating the best suggestions."
}}
Ensure the response is well-formed JSON and contains both required fields.

5. Handle Edge Cases:
    5.1. If the evaluation list is empty, return a default response indicating no feedback is available:
    5.2. If only one verifier provides output, use its feedback and suggestion directly, but rephrase for clarity if needed.
    5.3. If feedback or suggestions are vague or incomplete, infer reasonable conclusions based on the query context and available data.

6. Incorporate Query Context:
    6.1 Ensure the combined feedback and suggestion are grounded in the query’s intent and context, addressing the specific task or problem posed.

[Your Task]
Given the evaluations from the verifier models and the query context, generate a single JSON object containing the combined_feedback and combined_suggestion by synthesizing the inputs. Ensure the output is tailored to the query and leverages all relevant insights from the evaluations.
                """
                
                msg_list = [
                    {"role": "user", "content": synthesizer_prompt},
                ]

                synthesized_evaluation ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['combined_feedback', 'combined_suggestion'], 0.0)
                print(f"================ MAS Refinement Feedback ================\n", synthesized_evaluation['combined_feedback'])
                print(f"================ MAS Refinement Suggestion ================\n", synthesized_evaluation['combined_suggestion'])
                evaluation = [synthesized_evaluation]
                # break
            else:
                attempt = 100000
                    # break
            
            judge_path = os.path.join(args.save_dir, f"{expr_name}_{aw['name']}_{example_id}_full_response")
            with open(judge_path, 'w') as judge_file:
                judge_file.write(f'Question: {task_queue[0].content}\nIteration: {aw['name']}\nFull Response:{raw_results}')

            if global_use_oracle_verifier:
                acc_list = acc_oracle_verifier_list
            else:
                acc_list = acc_model_verifier_list


            if args.defer_verifier:
                fitness_str = bootstrap_confidence_interval([0.0])
                aw["acc"] = np.mean([0.0])

            else:
                fitness_str = bootstrap_confidence_interval(acc_list)
                aw["acc"] = np.mean(acc_list)

            aw["fitness"] = fitness_str
            aw["total_cost"] = get_global("global_COST_TOTAL")

            print(f"acc_list:", acc_list)
            print(f"mean acc_list:", np.mean(acc_list))
            print(f"bootstrap_confidence_interval: {fitness_str}")

            if 'swe_bench' in args.dataset:
                extracted_answer = final_reponse[0].split('\n\nAnswer:', 1)[-1].strip()
                if '<patch>' in extracted_answer:
                    extracted_answer = extract_xml(extracted_answer, 'patch').strip()   
            else:
                extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

            if '[TOO_HARD]' in extracted_answer:
                extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        
            memory.append({extracted_answer:fitness_str})
            print(f'save json to {mem_path}')
            with open(mem_path, 'w') as json_file:
                json.dump(memory, json_file, indent=4)

            # save results

            report_filename = os.path.join(args.save_dir, f'{expr_name}_{aw["name"]}_{example_id}_{args.option}_debug.html')
            converted_code_filename = os.path.join(args.save_dir, f'{expr_name}_{aw["name"]}_{example_id}_converted.py')
            print(f"Writing report to {report_filename}")
            # with open(report_filename, "w") as fh:
            #     fh.write(common.make_report(results))
            
            with open(converted_code_filename, "w") as fh:
                fh.write(next_solution['code'])    
            
            # metrics = results.metrics | {"score": results.score}
            # print('metrics: ',metrics)
            print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))

            with open(oracle_acc_result_path, "a+") as fh:
                fh.write(f'experiemnt {example_id}: 1 (initial {aw["name"]}): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')
            
            max_score = max(max_score, acc_oracle_verifier_list[0])
            
            if attempt == 100000:
                break
            
            
    end_time_ = time.time()
    total_time = end_time_ - start_time_        
    
    final_results.append({
        "example_id": example_id,
        "score": max_score,
        "total_time": total_time,
        "max_cost": get_global("global_COST_TOTAL")
    })
    
    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
            
    return acc_oracle_verifier_list[0], total_time, result_path
        
async def run_single_agent_baselines(args, expr_name, example_id, task_queue, meta_model, verifier_model, pattern = None):

    questions = get_global("global_questions")
    questions = questions[str(example_id)]
    global_node_model = get_global("global_node_model")
    
    cost_per_query = get_global("global_COST_TOTAL_per_query")
    cost_per_query[str(example_id)] = 0.0
    set_global("global_COST_TOTAL_per_query", cost_per_query)

    print(f"problem length: {len(questions)}")
    max_workers = min(len(questions), args.max_workers) if args.multiprocessing else 1

    if args.dataset == 'gpqa_diamond':
        task_queue = [Info(field_name, author, {"question": content.question, "choice1": content.choice1, "choice2": content.choice2, "choice3": content.choice3, "choice4": content.choice4}, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]
    else:
        task_queue = [Info(field_name, author, content, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]

    set_global("global_max_workers", max_workers)
    result_path = expr_name + f"{args.dataset}/{pattern}"
    expr_name = expr_name + f"{args.dataset}/{pattern}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"

    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue    
    set_global("global_task_queue", global_task_queue)

    next_solution_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_next_solution.json")
    msg_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_msg.json")
    mem_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_mem.json")
    file_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_archive.json")
    result_path = f'results/{args.dataset}/single_agent_baselines/{pattern}/{meta_model}_{global_node_model}_{verifier_model}.results'
    oracle_acc_result_path = f'results/{args.dataset}/single_agent_baselines/{pattern}/{meta_model}_{global_node_model}_oracle.results'
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
    
    final_results_path = f'results/{args.dataset}/single_agent_baselines/{pattern}/{meta_model}_{global_node_model}/final_results_{example_id}.json'
    result_path = Path(final_results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    final_results = []

    judge_path = os.path.join(args.save_dir, f"{expr_name}_{pattern}_{args.option}_judge")
    reponse_path = os.path.join(args.save_dir, f"{expr_name}_{pattern}_{args.option}_reponse")
    print(f"================== Test single agent baselines {pattern} ===================")
    default_global_n = get_global("global_n")
    default_global_n[str(example_id)] = f"Baseline {pattern}"
    set_global("global_n", default_global_n)

    global_n = get_global("global_n")
    global_n = global_n[str(example_id)]
    global_ns.append(global_n)
    
    blocks = get_init_archive(['cot', 'sc_cot', 'reflexion', 'debate'])

    workflow = blocks[pattern]

    try:
        workflow["code"] = workflow['code'].replace("forward", f"forward_{example_id}")
        acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, _, _, _, total_time = await evaluate_forward_fn(args, example_id, workflow["code"])
    except Exception as e:
        print("Error: ", str(e))
    
    judge_path = os.path.join(args.save_dir, f"{expr_name}_{pattern}_full_response")
    with open(judge_path, 'w') as judge_file:
        judge_file.write(f'Question: {task_queue[0].content}\Pattern: {pattern}\nFull Response:{raw_results}')
        
    with open(oracle_acc_result_path, "a+") as fh:
        fh.write(f'experiemnt {example_id}: 1 (initial Baseline_{pattern}): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')

    #TODO: can we somehow also log acc_oracle_verifier_list so that we can know how accurate acc_model_verifier_list is?
    if global_use_oracle_verifier:
        acc_list = acc_oracle_verifier_list
    else:
        acc_list = acc_model_verifier_list


    if args.defer_verifier:
        fitness_str = bootstrap_confidence_interval([0.0])
        workflow["acc"] = np.mean([0.0])

    else:
        fitness_str = bootstrap_confidence_interval(acc_list)
        workflow["acc"] = np.mean(acc_list)

    workflow["fitness"] = fitness_str
    workflow["total_cost"] = get_global("global_COST_TOTAL")

    print(f"acc_list:", acc_list)
    print(f"mean acc_list:", np.mean(acc_list))
    print(f"bootstrap_confidence_interval: {fitness_str}")


    if 'swe_bench' in args.dataset:
        extracted_answer = final_reponse[0].split('\n\nAnswer:', 1)[-1].strip()
        if '<patch>' in extracted_answer:
            extracted_answer = extract_xml(extracted_answer, 'patch').strip()   
    else:
        extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

    if '[TOO_HARD]' in extracted_answer: # we cannot add [TOO_HARD] in memory
        extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        
    memory.append({extracted_answer:fitness_str})
    print(f'save json to {mem_path}')
    with open(mem_path, 'w') as json_file:
        json.dump(memory, json_file, indent=4)
    final_results.append({
        "example_id": example_id,
        "score": acc_oracle_verifier_list[0],
        "total_time": total_time,
        "max_cost": get_global("global_COST_TOTAL")
    })

    # save results

    report_filename = os.path.join(args.save_dir, f'{expr_name}_Baseline_{pattern}_{args.option}_debug.html')
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(results))
    metrics = results.metrics | {"score": results.score}
    print('metrics: ',metrics)
    print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))
    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    return acc_oracle_verifier_list[0], total_time, result_path