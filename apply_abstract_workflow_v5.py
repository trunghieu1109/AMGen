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
from datetime import datetime

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
    Attributes:a
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
                    input_infos_text += f' Related original question:\n\n{content}.These are just pieces of information related to the question. You are not required to answer the question — just follow what is defined in the instruction: {instruction}.   \n\nRelated sub-task questions and answers:\n\n'
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

#                 prompt = input_infos_text + f'''
# Please think broadly, explore multiple dimensions (e.g., effectiveness, generalizability, popularity, clarity, efficiency, potential risks), and adapt your reasoning dynamically depending on the nature of the plans. You may adjust the evaluation criteria if some aspects prove more important than others.
# Task: Given the above, answer the following question: {instruction}\n\n.  Think deeply, follow the instruction step-by-step. Return your answer in the "answer" entry and justify detailedly how you could get this answer in "thinking" entry. Answer is a string include the answer for this query. If you require to return `feedback` and `correct`, just return the these fields.
#                 '''
                prompt = input_infos_text + f'''Given the above, answer the following question: {instruction} \n\n then justify completely and detailedly, step-by-step why you think so in the "thinking" entry. 
                Consider all cases that are possible to happen.
                Avoid some unclear explainations, such as "Using the circle equation or the condition for four points to be concyclic, we derive an equation in x. Solving this quadratic equation yields x=36.".
                In default, return response in json format.
                '''# instruction (sub-task in above)

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

        response_json ,_ = await get_json_response_from_gpt(prompt, self.model, self.output_fields, self.temperature, is_execution=True)

        output_infos = []
        cnt = 0
        for key, value in response_json.items():
            info = Info(key, await self.__repr__(), str(value), prompt, None, None, iteration_idx)
            output_infos.append(info)
            cnt += 1
            if cnt == len(self.output_fields):
                break
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

async def evaluate_forward_fn(args, example_id, forward_str, task_queue):
    # dynamically define forward()
    # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py


    print('forward_str: ', forward_str)

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
    results, logs = await call_forward(example_id, task_queue[0])
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
            
async def apply_abstract_workflow_enhance(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None, date_time=""):
    if not example_id in [157, 165, 175, 176, 180, 190]:
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
        
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/cluster_to_subtask_mapping.json', 'r', encoding='utf-8') as f:
        cluster_to_subtask = json.load(f)
        
    abstracted_subtask = [subtask for name, subtask in cluster_to_subtask.items()]
        
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
    oracle_acc_result_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_result_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{meta_model}_{global_node_model}_oracle.results'
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
    
    print(thinking1.content)
```

SC_CoT:
```python
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
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
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)
        
    # The most common answer is chosen for consistency and accuracy.
    final_instr = "Given all the above thinking and answers, find the most consistent and correct solutions for the [problem]"
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, 
                                                 "Sub-task 5: Synthesize and choose the most consistent answer for [problem]" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}}; answer - {{answer2.content}}")
    subtask_desc2['response'] = {{
        "thinking": thinking2,
        "answer": answer2
    }}
    logs.append(subtask_desc2)
    print(thinking2.content)
    
```

Reflexion:
```python
    reflect_inst =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Your problem is ... [problem]." + reflect_inst
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
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct"
    for i in range(N_max):
        # Critic agent debates and criticizes pros and cons of previous version
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst, 
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
    print(thinking3.content)
    
```

** Note **: For Reflexion pattern, you must add "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better." with the reflect_instruction and "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct" to the critic_instruction.


Debate:
```python
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_5 = "Sub-task 5: Your problem is .... [instruction]." + debate_instr
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    subtask_desc5 = {{
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"]
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
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {{agent.id}}, round {{r}}, converting [intermediate output] and calculating [final output], thinking: {{thinking5.content}}; answer: {{answer_5.content}}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    # Final decision agent makes final decision
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_thinking5[-1], 
                                                 "Sub-task 5: [problem]" + final_instr, 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating [final output], thinking: {{thinking5.content}}; answer: {{answer5.content}}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {{thinking5.content}}; answer - {{answer5.content}}")
    subtask_desc5['response'] = {{
        "thinking": thinking5,
        "answer": answer5
    }}
    logs.append(subtask_desc5)
    print(thinking5.content)
    
```
** Note **: For Debate patterns, you must add "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer." to the debate_instruction and "Given all the above thinking and answers, reason over them carefully and provide a final answer." to the final_decision_instruction.
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
    subtask_names = set()
    for aw in abstract_workflow:
        for stage_id, stage in aw['flow'].items():
            subtask_names.add(stage['Title'])
            
    os.makedirs(f"logs/{args.dataset}/{meta_model}", exist_ok=True)
    log_path = f"logs/{args.dataset}/{meta_model}/{date_time}_logs_{example_id}.txt"
    
    # decompose task to high-level subtasks
    potential_high_level_plan = []
    for i in range(0, 5):
        task_high_level_decomposition = f"""
Role and Responsibility:
You are an expert LLM assistant trained to analyze and decompose a user query into its core subtasks. Your goal is to:

- Identify and list essential subtasks that are necessary to accomplish the user’s input query.
- Each subtask must represent a distinct and meaningful action or process step that contributes directly to solving the overall task, do not analyze too specific to each details in query.
- These subtasks must follow the reasoning model, from subtasks 1 -> subtask 2.
- Remove many trivial steps that are not the core subtasks.
- Maximum 4 subtasks.
User query:
{task_queue[0].content}

Output Format Requirements:
You must return your result in valid JSON format with the following structure:

json
{{
    "thought": "Your thought while decomposing query",
    "subtask_list": [
        {{
            "objective": "Brief and clear description of the core subtask"
        }}
    ]
}}

Each objective must be a concise, clear, and self-contained description of a core subtask.

Do not include any additional fields or explanations outside of the objective key.
        """
        
        msg_list = [
            {"role": "user", "content": task_high_level_decomposition},
        ]

        high_level_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'subtask_list'], 0.0)
        
        print("high_level_decomposition: ", high_level_decomposition['subtask_list'])
        
        potential_high_level_plan.append(high_level_decomposition)
        
    final_high_level_decomposition_prompt = f"""
I have a set of candidate plans, each representing a possible solution to the same task. Please analyze all the given plans and select the most popular and best one based on the following criteria:
1. Popularity: If possible, consider how frequently each plan (or similar structure/idea) appears among the options.
2. Effectiveness: Which plan is most likely to succeed in accomplishing the intended task?
3. Clarity and feasibility: Is the plan clear, implementable, and realistic?

Potential High Level Plan:
{potential_high_level_plan}

Maximum 4 subtasks.

Output Format Requirements:
You must return your result in valid JSON format with the following structure:

json
{{
    "thought": "Your thought while decomposing query",
    "subtask_list": [
        {{
            "objective": "Brief and clear description of the core subtask"
        }}
    ]
}}

Each objective must be a concise, clear, and self-contained description of a core subtask.

Do not include any additional fields or explanations outside of the objective key.
    """
    
    msg_list = [
        {"role": "user", "content": final_high_level_decomposition_prompt},
    ]

    final_high_level_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'subtask_list'], 0.0)
    
        
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "high level task decomposition"
        content = final_high_level_decomposition['subtask_list']
        
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    high_level_subtasks = [subtask['objective'] for subtask in final_high_level_decomposition['subtask_list']]
    
    # embeddings, postprocess and clustering the embeddings
    abstractor = MASAbstraction()
    abstracted_subtasks = await abstractor.abstract_task_decomposition(task_queue[0].content, high_level_subtasks, subtask_names, abstracted_subtask)
    merged_subtasks = [f"{a_subtask['subtask_name']}: {a_subtask['abstracted_objective']}" for a_subtask in abstracted_subtasks]
    print(merged_subtasks)
    
    embeddings = await abstractor.embedding_subtask(merged_subtasks)
    
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/kmeans.pkl', 'rb') as f:
        kmeans = pkl.load(f)
    
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/pca.pkl', 'rb') as f:
        pca = pkl.load(f)
        
    normalized_embeddings = normalize(embeddings, norm="l2")
    reduced_embeddings = pca.transform(normalized_embeddings)
    
    cluster_id = kmeans.predict(reduced_embeddings)
    print("Clustering results: ", cluster_id)
    mas_chain = []
    for num_id, idx in enumerate(cluster_id):
        if str(idx) not in mas_chain or num_id == len(cluster_id) - 1:
            mas_chain.append(str(idx))
            
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

    print("workflow index: ", workflow_index)
    # return 1, 1, 1, ""
    
    filterd_workflow = [abstract_workflow[idx] for idx in workflow_index]
    
    print("Filtered workflow: ", [fw['name'] for fw in filterd_workflow])
    with open(filterd_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
        aw_desc = json.load(f)
    task_content = task_queue[0].content
    
    # decompose task to high-level subtasks
    task_detail_analysis = f"""
Provide a detailed analysis of the given problem by addressing the following points. Ensure the response is clear, logical, and uses appropriate mathematical notation. The analysis should be thorough yet concise, avoiding unnecessary computation unless requested.
Do not specify the objective / goal of this question.
1. Extract and Summarize Given Information:
    - Identify all key details provided in the problem, including numerical values, geometric objects, conditions, constraints.
    - Summarize the properties of any mathematical or geometric entities involved (e.g., shapes, dimensions, equations, or parameters).
2. Analyze Relationships Between Components:
    - Describe how the given entities or quantities are interconnected (e.g., geometric constraints, algebraic relationships, dependencies between variables).
    - Explain the significance of any conditions or constraints (e.g., tangency, equality, bounds) and how they influence the problem’s structure.
    - Hypothesize how the components might contribute to solving the problem or achieving the stated objective.
3. Identify the Field of Study:
    - Specify the mathematical domain(s) relevant to the problem (e.g., geometry, algebra, calculus, number theory, combinatorics).
    - Mention any subfields or specific concepts involved (e.g., 3D geometry, differential equations, graph theory).
    - Note potential applications or contexts where such problems arise (e.g., mathematical competitions, physics, computer science).
4. Highlight Aspects Needing Clarification:
    - Identify any ambiguous terms, conditions, or assumptions in the problem statement (e.g., undefined positions, unclear constraints).
    - List potential challenges in interpreting or solving the problem (e.g., complex computations, multiple possible configurations).

Note:
- Do not answer or suggest solution to solve the query. Only analysis information in it.
- Only extract the information from this query. 

User query:
{task_queue[0].content}

Output Format Requirements:
You must return your result in valid JSON format with the following structure:

json
{{
    "thought": "Your thought while decomposing query",
    "analysis": ""
}}

Do not include any additional fields or explanations outside of the objective key.
    """
    
    msg_list = [
        {"role": "user", "content": task_detail_analysis},
    ]

    detailed_analysis ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'analysis'], 0.0)
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "task analysis"
        content = detailed_analysis['analysis']
        
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    print(f"\n================== Query analysis: {detailed_analysis['analysis']}========================\n")
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, str(task_content) + "\n\nDetailed Analysis: \n" + str(detailed_analysis['analysis']), task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    output_description = get_global("global_output_description")
    max_attempt = 2
    acc_oracle_verifier_list = [0]
    total_time = 0
    final_results_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{meta_model}_{global_node_model}/final_results_{example_id}.json'
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
            # attempt = 1
            print(f'============Initial Archive: {aw["name"]}, attempt {attempt}=================')
            default_global_n = get_global("global_n")
            default_global_n[str(example_id)] = f"{aw["name"]}_{example_id}"
            set_global("global_n", default_global_n)

            global_n = get_global("global_n")
            global_n = global_n[str(example_id)]
            global_ns.append(global_n)
            
            user_prompt = ""
            dependencies_and_agent_collaboration = []
            for idx, st in enumerate(aw_desc):
                dependencies_and_agent_collaboration.append({
                    'stage_id': f'stage_{idx}',
                    'agent_collaboration': st['agent_collaboration'],
                    'dependencies': [f"From stage_{idx} to {dep.replace("subtask", "stage")}" for dep in st['dependencies']]
                })
            
            print(dependencies_and_agent_collaboration)
            
            # decompose query into multiple subtasks
            if attempt == 0:
                user_prompt = f"""
You are an agent specializing in task decomposition. Given a query, your job is to break it into subtasks with clear objectives and dependencies. 

[Inputs]
- Query: {task_queue_tmp[0].content}
- Abstract Workflow: {aw['flow']}
- Dependencies and Agent Collaboration: {dependencies_and_agent_collaboration}

[Instructions]
1. Do not decompose query to too small / trivial subtasks. Maximum 5 subtasks.
2. For each subtask, include:
- Objective: Objective that need to address in this subtask.
- Dependencies: Subtasks that must be completed before this one.
- Agent Collaboration Patterns: SC_CoT | Debate | Reflexion | CoT.  Thứ tự ưu tiên khi lựa chọn các patterns với khả năng reasoning mạnh, ví dụ như Debate > SC_CoT > Reflexion > CoT. 
3. A subtask can draw context from multiple previous steps to provide additional information for the current step, not just from the immediately preceding one. Therefore, try to choose appropriately. The dependencies are a useful reference: if there is a dependency from stage A to stage B, it means that steps in A can serve as context for steps in B.
5. If the output of a certain step can support or is related to a subtask, create a dependency between those subtasks
6. Outputs of one subtask should serve as proper inputs to the next.
7. Adhere to the structure defined by the Abstract Workflow to decompose the query into essential steps for solving the problem. Ensure strict compliance with the workflow and concretize each step using details extracted from the query.
8. Within each Stage, there may be multiple subtasks.
9. Dependencies should be established based on the Abstract Workflow structure as well as the intrinsic relationship between the query content and individual subtasks. For instance, if a subtask produces information A that is relevant or supportive to subsequent subtasks, a dependency should be explicitly defined between them.

[Output format — JSON]
{{
    "thought": "<your brief reasoning on how you break the query>",
    "task_decomposition": {{
        "stage_1": {{
            "subtask_1": {{
                "objective": "...",
                "dependencies": [],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }},
            "subtask_2": {{
                "objective": "...",
                "dependencies": ["subtask_1", ...],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }}
        }},
        "stage_2": {{
        ...
        }}
    }}
}}
                """
            else:
                failure_reason = [eva.get("failure_reason", "") for eva in evaluation]
                user_prompt = f"""
You are an agent specializing in task decomposition. The previous task decomposition encountered several issues that led to inefficiency. Therefore, please revise the task decomposition based on the evaluation results from earlier.

[Inputs]
- Query: {task_queue_tmp[0].content}
- Evaluation from previous attempts: {evaluation}
- Previous decomposition: {task_decomposition}
- Dependencies and Agent Collaboration: {dependencies_and_agent_collaboration}

[Instructions]
1. Do not decompose query to too small / trivial subtasks.
2. For each subtask, include:
- Objective: Objective that need to address in this subtask. Leverage feedback from previous attempts. Especially, embed the failure reason / feedback to enhance the objective of each subtask.
- Dependencies: Subtasks that must be completed before this one.
- Agent Collaboration Patterns: SC_CoT | Debate | Reflexion | CoT. Thứ tự ưu tiên khi lựa chọn các patterns với khả năng reasoning mạnh, ví dụ như Debate > SC_CoT > Reflexion > CoT. 
3. Use feedback (if any) to:
- Add những feedback về lỗi trong qua trình reasoning vào trong objective của subtasks để tránh lặp lại lỗi sai. Nguyên nhân bị sai: {failure_reason}
- Xem xét các gợi ý và feedback và sửa lại task decomposition:
    + Fix missing dependencies between subtasks.
    + Decompose subtasks to smaller subtasks.
    + Clarify unclear steps.
    + Refine objective of each subtasks: Revise them thoroughly to avoid repeating the mistakes highlighted in the experts' feedback.
4. subtask can draw context from multiple previous steps to provide additional information for the current step, not just from the immediately preceding one. Therefore, try to choose appropriately. The dependencies are a useful reference: if there is a dependency from stage A to stage B, it means that steps in A can serve as context for steps in B.
6. If the output of a certain step can support or is related to a subtask, create a dependency between those subtasks
7. Ensure better granularity, clarity, and logical flow than past attempts.
8. Ensure that all suggestions and feedback are integrated into the task decomposition by explicitly embedding them within the instructions of the corresponding subtasks.
9. Dependencies should be established based on the Abstract Workflow structure as well as the intrinsic relationship between the query content and individual subtasks. For instance, if a subtask produces information A that is relevant or supportive to subsequent subtasks, a dependency should be explicitly defined between them.

[Output format — JSON]
{{
    "thought": "<your brief reasoning on how you refined the breakdown>",
    "task_decomposition": {{
        "stage_1": {{
            "subtask_1": {{
                "objective": "...",
                "dependencies": [],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }},
            "subtask_2": {{
                "objective": "...",
                "dependencies": ["subtask_1", ....],
                "agent_collaboration": SC_CoT | Debate | Reflexion | CoT
            }}
        }},
        "stage_2": {{
        ...
        }}
    }}
}}
                """
                
            msg_list = [
                {"role": "user", "content": user_prompt},
            ]
 
            task_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'task_decomposition'], 0.0)
            print("\n============= Task Decomposition: =============\n", task_decomposition['task_decomposition'])
            stage_desc = str(aw).replace("subtask", "stage")
            with open(log_path, "a+", encoding="utf-8") as f:
        
                phase = f"task decomposition {attempt}"
                content = task_decomposition['task_decomposition']
                f.write(f"\n============== {phase} ================\n")
                f.write(str(content))
            # return 0, 0, 0, ""
            acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = [], [], [], '', '', "", [], [], "", "", ""

            user_prompt_generate_workflow = f"""
You are tasked with generating a concrete multi-stage agentic workflow using techniques like Chain-of-Thought, Self-Consistency, Reflexion, or Debate. Your goal is to implement the workflow for the provided query, following the task decomposition and improving from previous feedback.

Ensure this code is runnable. Double-check many times.

[Inputs]
1. Query: {task_queue_tmp[0].content}
2. Task Decomposition: {task_decomposition['task_decomposition']}
3. Abstract Workflow Template: {ABSTRACTED_WORKFLOW_TEMPLATE}. Return code in format of a function `forward`
4. Interaction Pattern Guide: {interaction_pattern}

[Requirements]
1. Format your output in JSON format as:
{{
  "thought": "Brief reasoning behind your workflow implementation.",
  "code": "Python code implementing the concrete agentic workflow."
}}
2. For each subtask:
   - Ensure that each subtask strictly follows the instruction defined in the task decomposition and incorporates relevant evaluation feedback and suggestions.
   - Apply the exactly agent collaboration patterns to each subtask as specified in task decomposition.
   - Pass the outputs from Subtask A to Subtask B in accordance with the defined dependencies between the subtasks.
   - Follow the task decomposition strictly (stage, order, dependencies).
   - Implement using appropriate agent collaboration pattern (CoT, SC-CoT, Reflexion, Debate).
   - Replace any placeholder with task-specific content grounded in the query.

3. Preserve logic from abstract workflow:
   - Use same number of stages.
   - Link subtasks clearly via outputs/inputs.
   - Synthesize final answer using `make_final_answer`.
   - Print each step using `print("Step x: ", sub_tasks[-1])`.
   - Include `from collections import Counter`.
   
5. Use variables: `self.node_model`, `self.debate_role`, `self.max_sc`, `self.max_round`.
6. Do not use LaTeX.
7. Do not include comments in code.

[Goal]
Ensure all subtasks are well-structured, grounded, and correctly executed. Output a fully working Python-based workflow.

            """
            
            msg_list = [
                {"role": "user", "content": user_prompt_generate_workflow},
            ]

            next_solution ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'code'], 0.0)
            print("================ Generated Multi-agent system ================\n", next_solution['code'])
            next_solution['code'] = next_solution['code'].replace("{{", "{")
            next_solution['code'] = next_solution['code'].replace("}}", "}")
            print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
            # evaluate multi agent system
            with open(log_path, "a+", encoding="utf-8") as f:
        
                phase = f"code generate {attempt}"
                content = next_solution['code']
                f.write(f"\n============== {phase} ================\n")
                f.write(str(content))
            try:
                next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
                # if 'gpqa_diamond' in args.dataset:
                #     analysis_only_task_queue = [Info(field_name, author, {"question": content.question, "choice1": content.choice1, "choice2": content.choice2, "choice3": content.choice3, "choice4": content.choice4}, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]
                # else:
                #     analysis_only_task_queue = [Info(field_name, author, content, prompt, sub_tasks, agnets, iteration_idx) for field_name, author, content, prompt, sub_tasks, agnets, iteration_idx in task_queue]
                acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"], task_queue_tmp)
                total_execution_time += total_time
            except Exception as e:
                print("Error: ", str(e))
                error_trace = traceback.format_exc()
                print("Full error trace:\n", error_trace)
                with open(f"error_mas/error_mas_{example_id}.py", "w") as f:
                    f.write(next_solution['code'])
                return -1, -1, -1, ""
            # return 1, 1, 1, ""
            with open(log_path, "a+", encoding="utf-8") as f:
        
                phase = f"results {attempt}"
                content = acc_oracle_verifier_list
                f.write(f"\n============== {phase} ================\n")
                f.write(str(content))
            
            # print("========== LOGS ===========\n", logs)
            
            subtask_desc = logs
            
            with open(log_path, "a+", encoding="utf-8") as f:
        
                phase = f"Response Text"
                content = logs
                f.write(f"\n============== {phase} ================\n")
                f.write(str(content))
 
            if not acc_oracle_verifier_list[0] and attempt < max_attempt - 1:
                # TODO refine this workflow
                # pass
                prev_context = []
                reasoning_refinement_prompt = f"""
You are a Verification Agent. Your task is to review the reasoning process of previous agents to find errors, check if contexts were sufficient, analyze how agents interacted, and suggest workflow improvements. The final answer is known to be wrong based on professor feedback — identify why and how to fix it.

# Input
1. User Query: {task_queue_tmp[0].content}
2. Subtasks:
   - Instruction
   - Context
   - Response (thinking + answer)
   - Final Answer: {current_ans}
3. Output Format: {output_description}
4. Reasoning Process of Previous Agents:
{subtask_desc}

# Tasks
1. Reasons of Failure:
   - Phân tích một cách chi tiết lý do tại sao mà reasoning process trước đó bị sai? Sai ở bước nào? Lý do tại sao lại sai?
    - Mô tả một cách rất chi tiết lỗi sai.
2. **Reasoning Check**:
   - Spot flawed logic or wrong assumptions in subtask reasoning.
   - Identify where the error happened that caused the final answer to fail.
   - Xác định cụ thể lỗi sai trong reasoning process (sai ở điểm suy luận nào đâu? tại sao lại sai? nên sửa như thế nào? sửa bằng cách nào?).
   - Nghiêm cấm đưa ra đáp án trực tiếp của câu hỏi trong feedback. 
   - Phân tích xem lỗi sai bắt nguồn từ bước nào?

3. **Context Evaluation**:
   - Was context for each subtask enough and relevant?
   - If not, what was missing or unclear?
   - Xem xét trong quá trình reasoning của previous agents và code của workflow. Suy nghĩ xem liệu rằng có thể đưa thêm context từ bước A -> bước B nào đó hay không?
   
4. **Agent Interaction Analysis**:
   - Did subtasks pass outputs correctly?
   - Was the collaboration pattern (e.g. CoT, Debate, Reflexion, SC CoT) effective?
   - Did poor interaction contribute to failure?
   - Do subtasks need to change the collaboration patterns? 
   - Tại các subtasks bị thất bại, xem xét việc chuyển đối agent collaboration patterns của subtasks sang một patterns mạnh mẽ hơn.

5. **Workflow Improvement**:
Base on feedback about reasoning process, propose potential improvements if necessary, such as:
    - Refine or break down subtasks that were failed.
    - Use better collaboration patterns for failed subtasks.
    - Improve instructions of failed subtasks.
    - Reconnect context between steps.
Only return from 1 - 2 core solutions, which directly affect to the reasoning process.

# Output
Return JSON with:
{{  
    "failure_reason": "The reason for why previous process was failed."
    "feedback": "Your detailed analysis: what failed, where, and why.",
    "suggestion": "Concrete steps to improve the workflow based on feedback"
}}

# Notes
- Be clear and concise.
- Call out correct subtasks too.
- Prioritize root cause fixes.
                """

                msg_list = [
                    {"role": "user", "content": reasoning_refinement_prompt},
                ]

                verifier_hub = [
                    'o4-mini',
                    'gpt-4.1-mini',
                ]
                
                tasks = [
                    (verifier_model, asyncio.create_task(
                        get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['suggestion', 'feedback', 'failure_reason'], 0.0)
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
                        "verifier_name": verifier_model,
                        "evaluation": mas_feedback
                    })
                    
                for ev in evaluation:
                    print(f"\n================= {ev['verifier_name']} ==================\n")
                    print(ev["evaluation"])
                    
                # return 1, 1, 1, ""
                    
#                 # print("\n=========== Evaluation ===========\n", evaluation)
            
#                 synthesizer_prompt = f"""
# You are an advanced agent specialized in synthesizing evaluations from multiple verifier models to produce a single, cohesive feedback and suggestion output. Your task is to analyze the feedback and suggestions provided by a set of verifiers, resolve any conflicts, and generate a unified feedback and suggestion that is clear, concise, and actionable. The input evaluations come from a list of verifier models (verifier_hub) and are stored in an evaluation list, where each entry contains feedback and suggestion fields.

# [Input Context]
# 1. Evaluations: A list of evaluation results, where each evaluation is a dictionary containing:
# - feedback: A string describing issues or observations about the query or task performance.
# - suggestion: A string proposing improvements or next steps.
# These feedback come from many verifiers:
# {evaluation}

# Query: The original query or task context: {task_queue_tmp[0].content}

# [Instructions]
# 1. Synthesize Output:
#     1.1. Combine the failure reason, avoiding redundancy.
#     1.1. Combine the feedback, avoiding redundancy.
#     1.2. Combine the suggestions that integrates the best ideas from all verifiers and addresses the query’s goals.    
#     1.3. Ensure the output is clear, avoids jargon unless necessary, and is structured for easy understanding.
# The Synthesized output must be detailed (failure in which subtasks?, why it was failed?, does the problem come from reasoning process?, does the problem come from agent collaboration patterns?)
# Provide a bullet-point formatted list including all the feedbacks and suggestions from experts.
# About the suggestions, for each of them, describe detailed how to modify.
# 2. Output Format:
# Return the result in JSON format with the following structure:
# {{
#     "combined_failure_reason": "The reason why previous process was failed"
#     "combined_feedback": "Issues or observations from all verifiers that are limitations of current workflow.",
#     "combined_suggestion": "A single, actionable recommendation integrating the best suggestions."
# }}
# Ensure the response is well-formed JSON and contains both required fields.

# 3. Incorporate Query Context:
#     3.1 Ensure the combined feedback and suggestion are grounded in the query’s intent and context, addressing the specific task or problem posed.

# [Your Task]
# Given the evaluations from the verifier models and the query context, generate a single JSON object containing the combined_feedback and combined_suggestion by synthesizing the inputs. Ensure the output is tailored to the query and leverages all relevant insights from the evaluations.
#                 """
                
#                 msg_list = [
#                     {"role": "user", "content": synthesizer_prompt},
#                 ]

#                 synthesized_evaluation ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['combined_failure_reason', 'combined_feedback', 'combined_suggestion'], 0.0)
#                 print(f"================ MAS Refinement Feedback ================\n", synthesized_evaluation['combined_feedback'])
#                 print(f"================ MAS Refinement Failure Reason ================\n", synthesized_evaluation['combined_failure_reason'])
#                 print(f"================ MAS Refinement Suggestion ================\n", synthesized_evaluation['combined_suggestion'])
#                 evaluation = [synthesized_evaluation]
                with open(log_path, "a+", encoding="utf-8") as f:
        
                    phase = f"evaluation {attempt}"
                    content = evaluation
                    f.write(f"\n============== {phase} ================\n")
                    f.write(str(content))
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
            
            # break
            
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
        
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = f"Evaluation Result"
        content = {
            "example_id": example_id,
            "score": max_score,
            "total_time": total_time,
            "total_execution_time": total_execution_time,
            "max_cost": get_global("global_COST_TOTAL"),
            "max_execution_cost": get_global("global_COST_EXECUTION")
        }
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
            
    return acc_oracle_verifier_list[0], total_time, total_execution_time, result_path
        
# async def run_single_agent_baselines(args, expr_name, example_id, task_queue, meta_model, verifier_model, pattern = None):

#     # if example_id < 150:
#     #     return 0, 0, 0, ""

#     questions = get_global("global_questions")
#     questions = questions[str(example_id)]
#     global_node_model = get_global("global_node_model")
    
#     cost_per_query = get_global("global_COST_TOTAL_per_query")
#     cost_per_query[str(example_id)] = 0.0
#     set_global("global_COST_TOTAL_per_query", cost_per_query)

#     print(f"problem length: {len(questions)}")
#     max_workers = min(len(questions), args.max_workers) if args.multiprocessing else 1

#     if 'gpqa_diamond' in args.dataset:
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
#     result_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{pattern}/{meta_model}_{global_node_model}_{verifier_model}.results'
#     oracle_acc_result_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{pattern}/{meta_model}_{global_node_model}_oracle.results'
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
    
#     final_results_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{pattern}/{meta_model}_{global_node_model}/final_results_{example_id}.json'
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
#         workflow["code"] = workflow['code'].replace("forward", f"forward_{example_id}")
#         acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, _, _, _, total_time = await evaluate_forward_fn(args, example_id, workflow["code"])
#     except Exception as e:
#         print("Error: ", str(e))
    
#     judge_path = os.path.join(args.save_dir, f"{expr_name}_{pattern}_full_response")
#     with open(judge_path, 'w') as judge_file:
#         judge_file.write(f'Question: {task_queue[0].content}\Pattern: {pattern}\nFull Response:{raw_results}')
        
#     with open(oracle_acc_result_path, "a+") as fh:
#         fh.write(f'experiemnt {example_id}: 1 (initial Baseline_{pattern}): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')

#     #TODO: can we somehow also log acc_oracle_verifier_list so that we can know how accurate acc_model_verifier_list is?
#     if global_use_oracle_verifier:
#         acc_list = acc_oracle_verifier_list
#     else:
#         acc_list = acc_model_verifier_list


#     if args.defer_verifier:
#         fitness_str = bootstrap_confidence_interval([0.0])
#         workflow["acc"] = np.mean([0.0])

#     else:
#         fitness_str = bootstrap_confidence_interval(acc_list)
#         workflow["acc"] = np.mean(acc_list)

#     workflow["fitness"] = fitness_str
#     workflow["total_cost"] = get_global("global_COST_TOTAL")

#     print(f"acc_list:", acc_list)
#     print(f"mean acc_list:", np.mean(acc_list))
#     print(f"bootstrap_confidence_interval: {fitness_str}")


#     if 'swe_bench' in args.dataset:
#         extracted_answer = final_reponse[0].split('\n\nAnswer:', 1)[-1].strip()
#         if '<patch>' in extracted_answer:
#             extracted_answer = extract_xml(extracted_answer, 'patch').strip()   
#     else:
#         extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

#     if '[TOO_HARD]' in extracted_answer: # we cannot add [TOO_HARD] in memory
#         extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        
#     memory.append({extracted_answer:fitness_str})
#     print(f'save json to {mem_path}')
#     with open(mem_path, 'w') as json_file:
#         json.dump(memory, json_file, indent=4)
#     final_results.append({
#         "example_id": example_id,
#         "score": acc_oracle_verifier_list[0],
#         "total_time": total_time,
#         "max_cost": get_global("global_COST_TOTAL")
#     })

#     # save results

#     report_filename = os.path.join(args.save_dir, f'{expr_name}_Baseline_{pattern}_{args.option}_debug.html')
#     print(f"Writing report to {report_filename}")
#     with open(report_filename, "w") as fh:
#         fh.write(common.make_report(results))
#     metrics = results.metrics | {"score": results.score}
#     print('metrics: ',metrics)
#     print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))
#     with open(final_results_path, "w") as f:
#         json.dump(final_results, f, indent=4)
    
#     return acc_oracle_verifier_list[0], total_time, 0, result_path


async def recheck_mas(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None):

    if example_id < 25 or example_id > 25:
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
    oracle_acc_result_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_result_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{meta_model}_{global_node_model}_oracle.results'
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
    
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, task_queue[0].content, task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    output_description = get_global("global_output_description")
    max_attempt = 2
    acc_oracle_verifier_list = [0]
    total_time = 0
    final_results_path = f'results/{args.dataset}/dev19_generation_model_test_v2/{meta_model}_{global_node_model}/final_results_{example_id}.json'
    result_path = Path(final_results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_results = []
    max_score = 0
    next_solution = {
        'code': ""
    }
    
    code_path = f"results/abstracted_based_same_model/question/meta_agent/workflow_search/aime24/{example_id}"
    model = "o4-mini"
    
    for filename in os.listdir(code_path):
        if filename.startswith(model) and filename.endswith(".py"):
            file_path = os.path.join(code_path, filename)
            with open("results/dev7/question/meta_agent/workflow_search/aime24/25/gpt-4.1-mini_gpt-4.1-mini_gpt-4.1-mini_abstracted_workflow_desc_2_25_converted.py", "r", encoding="utf-8") as f:
                next_solution['code'] = f.read()
                print(f"Nội dung của {filename}:\n")
                # print(content)
            break 
    else:
        print("Không tìm thấy file Python nào bắt đầu bằng 'o4_mini' và kết thúc bằng '.py'")
    
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
        