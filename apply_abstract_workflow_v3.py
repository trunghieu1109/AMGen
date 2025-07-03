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
from prompts.abstract_based_prompt import INTERACTION_PATTERN
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
    cot_agent, thinking1, answer1, subtask_desc1 = await self.cot("subtask_1", cot_instruction, [taskInfo], ['thinking', 'answer'], 0.0, "user input"):
    agents.append(f"CoT agent {{cot_agent.id}}, analyzing [expression #1], thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    logs.append(subtask_desc1)

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
    
    async def cot(self, subtask_id, cot_agent_desc):
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=cot_agent_desc['temperature'])
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_agent_desc['instruction'],
            "context": cot_agent_desc['context'],
            "agent_collaboration": "CoT"
        }
        thinking, answer = await cot_agent(cot_agent_desc['input'], subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        return {
            'cot_agent': cot_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }
    
    async def sc_cot(self, subtask_id, cot_sc_desc, n_repeat):
        cot_agents = [LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=cot_sc_desc['temperature']) for _ in range(n_repeat)]
        possible_answers = []
        thinkingmapping = {}
        answermapping = {}
        list_thinking = []
        list_answer = []
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_sc_desc['instruction'],
            "context": cot_sc_desc['context'],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(n_repeat):
            # Each CoT-SC agent tries to calculate all possible cases independently
            thinking, answer = await cot_agents[i](cot_sc_desc['input'], subtask_desc['instruction'], is_sub_task=True)
            possible_answers.append(str(answer.content))
            thinkingmapping[str(answer.content)] = thinking
            answermapping[str(answer.content)] = answer
            list_thinking.append(thinking)
            list_answer.append(answer)
            
        # The most common answer is chosen for consistency and accuracy.
        answer_content = Counter(possible_answers).most_common(1)[0][0]
        thinking = thinkingmapping[str(answer_content)]
        answer = answermapping[str(answer_content)]
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        return {
            'cot_agent': cot_agents,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc,
            'list_thinking': list_thinking,
            'list_answer': list_answer
        }
    
    async def reflexion(self, subtask_id, cot_reflect_desc, critic_desc, n_repeat):
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", model=self.node_model, temperature=cot_reflect_desc['temperature'])
        critic_agent = LLMAgentBase(['feedback', 'correct'], "Critic Agent", model=self.node_model, temperature=critic_desc['temperature'])
        
        # Input for CoT agent
        cot_inputs = cot_reflect_desc['input']
        
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_reflect_desc['instruction'],
            "context": cot_reflect_desc['context'],
            "agent_collaboration": "Reflexion"
        }
        
        feedbacks, corrects = [], []
        thinkings, answers = [], []
        
        # Generate the first version
        thinking, answer = await cot_agent(cot_inputs, cot_reflect_desc['instruction'], 0, is_sub_task=True)
        thinkings.append(thinking)
        answers.append(answer)
        for i in range(n_repeat):
            # Critic agent debates and criticizes pros and cons of previous version
            feedback, correct = await critic_agent(cot_reflect_desc['input'] + [thinking, answer], 
                                        critic_desc['instruction'], i, is_sub_task=True)
            feedbacks.append(feedback)
            corrects.append(correct)
            if correct.content == "True":
                break
            
            cot_inputs.extend([thinking, answer, feedback])
            thinking, answer = await cot_agent(cot_inputs, cot_reflect_desc['instruction'], i + 1, is_sub_task=True)
            thinkings.append(thinking)
            answers.append(answer)
            
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        print(f"Subtask {subtask_id} answer: thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'cot_agent': cot_agent,
            'critic_agent': critic_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc,
            'list_thinking': thinkings,
            'list_answer': answers,
            'list_feedback': feedbacks,
            'list_correct': corrects
        }
    
    async def debate(self, subtask_id, debate_desc, final_decision_desc, n_repeat):
        debate_agents = [LLMAgentBase(['thinking', 'answer'], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=debate_desc['temperature']) 
                        for role in self.debate_role]

        all_thinking = [[] for _ in range(n_repeat)]
        all_answer = [[] for _ in range(n_repeat)]
        
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": debate_desc['instruction'],
            "context": debate_desc['context'],
            "agent_collaboration": "Debate"
        }
        
        if final_decision_desc == None:
            final_decision_desc = {
                'output': ["thinking", "answer"],
                'instruction': debate_desc["instruction"],
                'temperature': 0.0
            }
        
        for r in range(n_repeat):
            # N_max_5 rounds of debating
            for i, agent in enumerate(debate_agents):
                # Each agent proposes its solution
                if r == 0:
                    thinking, answer = await agent(debate_desc['input'], 
                                            debate_desc['instruction'], r, is_sub_task=True)
                else:
                    # Generate next solution based on comments and counter-arguments from other debaters
                    input_infos = debate_desc['input'] + all_thinking[r-1] + all_answer[r-1]
                    thinking, answer = await agent(input_infos, debate_desc['instruction'], r, is_sub_task=True)
                
                all_thinking[r].append(thinking)
                all_answer[r].append(answer)
        
        # Final decision agent makes final decision
        final_decision_agent = LLMAgentBase(['thinking', 'answer'], "Final Decision Agent", 
                                            model=self.node_model, temperature=final_decision_desc['temperature'])
        thinking, answer = await final_decision_agent(debate_desc['input'] + all_thinking[-1] + all_answer[-1], 
                                                    final_decision_desc['instruction'], 
                                                    is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        print(f"Subtask {subtask_id} answer: thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'debate_agent': debate_agents,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc,
            'list_thinking': all_thinking,
            'list_answer': all_answer,
        }
        
    async def answer_generate(self, subtask_id, cot_agent_desc):
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Answer Generate Agent", model=self.node_model, temperature=cot_agent_desc['temperature'])
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_agent_desc['instruction'] + "\nThink step by step and solve the problem. Ensure the final answer is concisely and clearly, direct response to the question without including explanations or reasoning",
            "context": cot_agent_desc['context'],
            "agent_collaboration": "AnswerGenerate"
        }
        
        thinking, answer = await cot_agent(cot_agent_desc['input'], subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        return {
            'cot_agent': cot_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }
        
    async def specific_format(self, subtask_id, formatter_desc):
        formatter_agent = LLMAgentBase(['thinking', 'answer'], "SpecificFormatter Agent", self.node_model, temperature=formatter_desc['temperature'])
       
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": formatter_desc['instruction'] + f"\nExtract the correct answer for the question from the previous context. Then, return the final answer in the following format: {formatter_desc['format']}",
            "context": formatter_desc['context'],
            "agent_collaboration": "SpecificFormat"
        }
        
        thinking, answer = await formatter_agent(formatter_desc['input'], subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        return {
            'formatter_agent': formatter_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }
        
    async def aggregate(self, subtask_id, aggregate_desc):
        aggregate_agent = LLMAgentBase(['thinking', 'answer'], "Aggregate Agent", self.node_model, temperature=aggregate_desc['temperature'])
       
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": aggregate_desc['instruction'] + f"\nCarefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution",
            "context": aggregate_desc['context'],
            "agent_collaboration": "AggregateAgent"
        }
        
        thinking, answer = await formatter_agent(aggregate_desc['input'], subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        return {
            'aggregate_agent': aggregate_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }
        
    async def code_generate(self, subtask_id, code_generate_desc):
        
        code_generate_agent = LLMAgentBase(['thinking', 'code'], "Code Generate Agent", self.node_model, temperature=code_generate_desc['temperature'])
       
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": code_generate_desc['instruction'] + f"""
\nWrite complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.
The entry point of the function: {code_generate_desc['entry_point']}.
Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
                """,
            "context": code_generate_desc['context'],
            "agent_collaboration": "CodeGenerate"
        }
        
        thinking, code = await code_generate_agent(code_generate_desc['input'], subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = { 
            "thinking": thinking,
            "code": code
        }
        
        return {
            'code_generate_agent': code_generate_agent,
            'thinking': thinking,
            'code': code,
            'subtask_desc': subtask_desc
        }


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

async def high_level_task_decomposition(meta_model, query):
    task_high_level_decomposition = f"""
You are an expert LLM assistant trained to decompose user queries to only many core subtasks list. 
Your task is to decompose user queries to only many core subtasks list.

Maximum 4 subtasks.

Input Query:
{query}

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
    
    return high_level_decomposition

async def specific_task_decomposition(meta_model, query, output_description, aw_flow, collaboration_dependencies):
    user_prompt = f"""
You are an agent specialized in task decomposition. Your task is to analyze the provided query thoroughly and decompose it into subtasks and their dependencies to address the query effectively. You must incorporate feedback from previous decomposition attempts (evaluation) to improve the quality of the decomposition, avoiding past mistakes and addressing any identified issues.
[Query]
{query}

[Output format]
{output_description}

[Instruction]
1. Decompose the query into fine-grained, detailed subtasks. Ensure the subtasks are simple, specific, and aligned with the model's capabilities, avoiding overly complex steps.
2. Structure the subtasks in a multi-turn reasoning sequence (e.g., step 1 → step 2), ensuring logical progression.
3. For each subtask, include:
    3.1. Objective: Clearly state the purpose of the subtask, incorporating all relevant context or conditions from previous subtasks to enhance performance.
    3.2. Dependencies: Specify which subtasks must be completed before the current subtask, ensuring proper sequencing.
4. Follow the provided workflow description to structure the decomposition process.
5. Ensure the decomposition is tailored to the query’s intent and context.
6. Consider output format of each subtask, so that they could pass their output to other agents effectively.
7. Consider the suggested agent collaboration and subtask dependencies for each subtask.
8. Subtask id must start from 1 to n, eg. 1, 2, 3, 4, 5, ...
9. Subtask name in this format: "Subtask + subtask_id", subtask_id ~ 0, 1, 2, 3, 4, ...
10. Ensure output format of the workflow must match to output format requirement: {output_description}

<Abstract Workflow Description>
{aw_flow}
</Abstract Workflow Description>

<Agent Collaboration and Subtask Dependencies>
{collaboration_dependencies}

Return in JSON format, contains 'task_decomposition'. 
{{
    'thought': "Your thought while decomposing task -> to subtasks",
    'task_decomposition': {{
        'stage_1': {{
            'subtask_1': {{
                'objective': "",
                'agent_collaboration': [CoT | SC_CoT | Debate | Reflexion],
                'dependencies': []
            }},
            'subtask_2': {{
                'objective': "",
                'agent_collaboration': [CoT | SC_CoT | Debate | Reflexion],
                'dependencies': [subtask_1]
            }}
        }},
        'stage_2': {{
            'subtask_3': {{
                'objective': "",
                'agent_collaboration': [CoT | SC_CoT | Debate | Reflexion],
                'dependencies': [...]
            }},
            'subtask_4': {{
                'objective': "",
                'agent_collaboration': [CoT | SC_CoT | Debate | Reflexion],
                'dependencies': [...]
            }}
        }}
    }}
}}
    """

    msg_list = [
        {"role": "user", "content": user_prompt},
    ]

    task_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'task_decomposition'], 0.0)
    for k, v in task_decomposition['task_decomposition'].items():
        for ks, vs in v.items():
            vs['objective'] = vs['objective'].replace("{", "")
            vs['objective'] = vs['objective'].replace("}", "")
            vs['objective'] = vs['objective'].replace("'", "")
            
    return task_decomposition
    
async def generate_concretized_workflow(meta_model, task_decomposition, output_description, interaction_pattern, stage_desc, query):
    user_prompt_generate_workflow = f"""
You are tasked with instantiating a concrete version of an abstract multi-stage workflow to solve complex queries using agent-based reasoning techniques (e.g., Chain-of-Thought, Self-Consistency, Reflexion, Debate). The workflow must be customized for the provided query and incorporate feedback from previous attempts to improve contextual usage, reasoning processes, and overall effectiveness.

[Important Notes]
1. You may adapt the agent interaction patterns (e.g., CoT, Self-Consistency CoT, Debate, Reflexion) to best suit the query’s complexity and requirements.
2. The generated code must include sufficient subtasks as planned in the task decomposition.
3. A stage may include multiple steps if necessary to achieve the query’s objectives.
4. The final answer must be processed by `make_final_answer` function.
5. Subtask id must start from 1 to n, eg. 1, 2, 3, 4, 5, ...
6. Subtask name in this format: "Subtask + subtask_id", subtask_id ~ 0, 1, 2, 3, 4, ...
7. Ensure output format of the workflow must match to output format requirement: {output_description}
8. Follow strictly the design of defined interaction patterns.
9. You can only use `self.cot`, `self.sc_cot`, `self.debate` and `self.reflexion`.

[Instructions]
You must follow the requirements below when generating the concrete workflow:
1. Return Format
Reply EXACTLY in the following JSON format. The generated code MUST be in function format (i.e., forward function) without comments.
{{
    "thought": "Your thought while generating code",
    "code": "Your code."
}}
You must remove commnets from generated code.
Ensure the response is a well-formed JSON object and includes all required fields.

2. Workflow Decomposition
2.1. Follow the provided task decomposition to structure the workflow. 
[Task Decomposition] 
{task_decomposition}

3. Agent Collaboration Patterns
    3.1. Implement agent interaction structures such as Chain-of-Thought (CoT), Self-Consistency CoT (SC-CoT), Reflexion, or Debate, selecting the most appropriate pattern for each subtask based on its complexity and the query’s requirements.
    3.2. Retain variables like self.node_model, self.debate_role, self.max_sc, and self.max_round with their correct roles.
    3.3. Refer to the provided interaction pattern examples for implementation guidance. 
    3.4. You can only use `self.cot`, `self.sc_cot`, `self.debate` and `self.reflexion`.
    3.5. Use correct implementation of interaction patterns.
[Interaction Pattern] {interaction_pattern}

4. Placeholder Replacement
4.1. Replace placeholders (e.g., [final answer], [condition #1]) with specific information relevant to the subtask. For example, if the query involves calculating the speed of a nucleus, replace [final answer] with “the speed of the nucleus.”
4.2. Ensure reasoning goals are fully grounded in the query’s context for clarity and relevance.

Note:
- The final answer must be processed by `make_final_answer` function.
- Subtask id must start from 1 to n, eg. 1, 2, 3, 4, 5, ...

[Your Task]
Generate a concrete agentic workflow in Python code format, based on the Abstract Workflow, tailored to the query and informed by the evaluation feedback.

[Workflow Description]
{stage_desc}

[Workflow Template]
{ABSTRACTED_WORKFLOW_TEMPLATE}

[Query]
{query}
    """
    
    msg_list = [
        {"role": "user", "content": user_prompt_generate_workflow},
    ]

    next_solution ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'code'], 0.0)
    print("================ Generated Multi-agent system ================\n", next_solution['code'])
    next_solution['code'] = next_solution['code'].replace("{{", "{")
    next_solution['code'] = next_solution['code'].replace("}}", "}")
    
    return next_solution
    
async def evaluate_workflow(query, subtask_desc, current_ans, output_description, verifier_hub):
    reasoning_refinement_prompt = f"""
# Task: Diagnose and Improve Agentic Workflow Output

Based on feedback from professors, the current answer generated for the user query is incorrect.

You must:
- Identify the **root causes** of this failure.
- Provide **subtask-level analysis**.
- Propose **clear, structured suggestions** to improve the workflow’s correctness and performance.

---

## 🔹 Input

You will receive the following:

- **[User Query]**
  {query}

- **[Subtask Description]**
  {subtask_desc}

- **[Current Answer]**
  {current_ans}

- **[Output Format]**
  {output_description}

---

## 🔹 Your Responsibilities

Please carry out the following analyses:

### 1. Subtask-Level Reasoning Verification
For **each subtask**, perform:
- Review of its reasoning steps.
- Identification of whether the result is correct or incorrect.
- If incorrect, explain:
  - What went wrong (logical flaw, faulty assumption, etc.).
  - Why it caused failure.
- Provide a concrete recommendation to fix it.

### 2. Subtask-Level Context Sufficiency Evaluation
For each subtask:
- Assess whether its input context is sufficient.
- Identify missing, ambiguous, or misleading information.
- Suggest how to improve the context to ensure the agent performs the task more accurately.

### 3. Agent Interaction Analysis
- Agents in this workflow are represented in via any human-designed interaction patterns (e.g CoT (Chain-of-Thought), SC_CoT (Self Consistency CoT), Reflexion, Debate, v.v). So based on the reasoning process of these subtasks, are the interaction patterns of agents in this subtask effective?
- Determine if poor communication or lack of information sharing contributed to failure.
- Suggest how agents could collaborate more effectively.

### 4. Collaboration Pattern Recommendation
For each subtask, recommend the most appropriate reasoning pattern:
- **CoT (Chain-of-Thought)** – Linear, step-by-step thinking
- **SC-CoT (Self-Consistent CoT)** – Multiple reasoning paths with voting
- **Debate** – Agents challenge each other’s ideas
- **Reflexion** – Agent reviews and self-corrects errors

Briefly justify your recommendation per subtask.

### 5. Output Validation
- Check that the outputs conform to the required format.
- Assess the factual accuracy, completeness, and correctness.
- Point out any formatting mistakes, inconsistencies, or gaps in content.
- Verify whether output format of the workflow match to output format requirement: {output_description} or not.

### 6. Workflow Improvement Suggestions
Propose targeted enhancements to the workflow:
- **Decompose Subtasks** – Further break down overly complex subtasks.
- **Adjust Collaboration Pattern** – Use more suitable reasoning strategies.
- **Rewrite Instructions** – Make task prompts more explicit and unambiguous.
- **Refine Context Flow** – Ensure information flows smoothly between subtasks.

---

## Notes:


## 🔹 Expected Output

Return results in the following **JSON format** contains `feedback` and `suggestion` fields:

```json
{{
  "feedback": [
    {{
      "subtask": "Subtask 1",
      "is_correct": false,
      "issue": "The reasoning step assumes X without evidence, leading to incorrect conclusion Y.",
      "improvement": ["Clarify the premise or provide explicit context about X to guide the agent."]
    }},
    {{
      "subtask": "Subtask 2",
      "is_correct": true,
      "issue": null,
      "improvement": [null]
    }},
    ...
  ],
  "suggestion": [
    {{
      "subtask": "Subtask 1",
      "recommendation": {{
        "collaboration_pattern": "Reflexion",
        "justification": ["The agent made a subtle reasoning error that could be caught by reflecting on its own output."],
        "prompt_update": "Add a final step asking the agent to validate its answer against the original question."
      }}
    }},
    ...
  ]
}}
    """
    
    msg_list = [
        {"role": "user", "content": reasoning_refinement_prompt},
    ]
    
    tasks = [
        (verifier_model, asyncio.create_task(
            get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['suggestion', 'feedback'], 0.0)
        ))
        for verifier_model in verifier_hub
    ]
    
    # Run all tasks in parallel and gather results
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    evaluation = []
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
    
    return evaluation

async def synthesize_evaluations(verifier_model, evaluation, query):
    synthesizer_prompt = f"""
Your task is to analyze the feedback and suggestions provided by a set of verifiers, resolve any conflicts, and generate a unified feedback and suggestion that is clear, concise, and actionable. 
The input evaluations come from a list of verifier models (verifier_hub) and are stored in an evaluation list, where each entry contains feedback and suggestion fields.

[Input Context]
These feedback come from many verifiers:
{evaluation}

Query: The original query or task context: {query}

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
For each distinct subtask:
    3.1. Explain why it failed (if incorrect): reasoning flaw, lack of context, poor agent interaction, etc.
    3.2. Provide consolidated improvement suggestions.
    
Workflow Improvement Suggestions
Propose targeted enhancements to the workflow:
- **Decompose Subtasks** – Further break down overly complex subtasks.
- **Adjust Collaboration Pattern** – Use more suitable reasoning strategies.
- **Rewrite Instructions** – Make task prompts more explicit and unambiguous.
- **Refine Context Flow** – Ensure information flows smoothly between subtasks.


Recommend the best collaboration pattern and how to modify the prompt or structure accordingly.
The Synthesized output must be detailed (failure in which subtasks?, why it was failed?, does the problem come from reasoning process?, does the problem come from agent collaboration patterns?)
About the suggestions, for each of them, describe detailed how to modify.
4. Output Format
Return the output in the following JSON format:
{{
  "combined_feedback": [
    {{
      "subtask": "Subtask 1",
      "is_correct": false,
      "issue": "The reasoning step assumes X without evidence, leading to incorrect conclusion Y.",
      "improvement": ["Clarify the premise or provide explicit context about X to guide the agent."]
    }},
    {{
      "subtask": "Subtask 2",
      "is_correct": true,
      "issue": null,
      "improvement": [null]
    }},
    ...
  ],
  "combined_suggestion": [
    {{
      "subtask": "Subtask 1",
      "recommendation": {{
        "collaboration_pattern": "Reflexion",
        "justification": ["The agent made a subtle reasoning error that could be caught by reflecting on its own output."],
        "prompt_update": "Add a final step asking the agent to validate its answer against the original question."
      }}
    }},
    ...
  ]
}}
✅ Ensure your final JSON is valid and complete.

[Your Task]
Given the evaluations from the verifier models and the query context, generate a single JSON object containing the combined_feedback and combined_suggestion by synthesizing the inputs. Ensure the output is tailored to the query and leverages all relevant insights from the evaluations.
    """

    msg_list = [
        {"role": "user", "content": synthesizer_prompt},
    ]

    synthesized_evaluation ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['combined_feedback', 'combined_suggestion'], 0.0)
    print(f"================ MAS Refinement Feedback ================\n", synthesized_evaluation['combined_feedback'])
    print(f"================ MAS Refinement Suggestion ================\n", synthesized_evaluation['combined_suggestion'])
    
    return synthesized_evaluation

async def refined_task_decomposition(meta_model, query, output_description, synthesized_evaluation, prev_task_decomposition):
    user_prompt = f"""
You are an agent specialized in task decomposition. Your task is to analyze the provided query thoroughly and decompose it into subtasks and their dependencies to address the query effectively. You must incorporate feedback from previous decomposition attempts (evaluation) to improve the quality of the decomposition, avoiding past mistakes and addressing any identified issues.
[Query]
{query}

[Output format]
{output_description}

[Instruction]
1. Decompose the query nto fine-grained, detailed subtasks. Ensure the subtasks are simple, specific, and aligned with the model's capabilities, avoiding overly complex steps.
2. Each subtask must be corresponding to only one single objective. Do not group multiple objective in a subtask.
3. Structure the subtasks in a multi-turn reasoning sequence (e.g., step 1 → step 2), ensuring logical progression.
4. For each subtask, include:
    4.1. Objective: Clearly state the purpose of the subtask, incorporating all relevant context or conditions from previous subtasks to enhance performance.
    4.2. Dependencies: Specify which subtasks must be completed before the current subtask, ensuring proper sequencing.
5. Analyze the evaluation from previous decomposition attempts to identify weaknesses, redundancies, or gaps. Use this feedback to refine the decomposition, ensuring:
    5.1. Subtasks address any issues highlighted in the evaluation (e.g., overly broad tasks, missing dependencies, or unclear objectives).
    5.2. The decomposition improves upon previous attempts by being more precise, efficient, or comprehensive.
6. Incorporate domain expert feedback:
    6.1. Carefully review all feedback and evaluation from previous attempts.
    [Evaluation]
    {synthesized_evaluation}
    6.2. Leverage any suggestions or expert critique to address prior issues such as:
        6.2.1. Missing dependencies.
        6.2.2. Overly broad or abstract subtasks.
        6.2.3. Redundant steps.
        6.2.4. Gaps in logic or execution order.
7. Refine the current decomposition by ensuring:
    7.1. Improved granularity, clarity, and logical sequencing over previous attempts.
    7.2. Outputs from one subtask are well-formed and suitable as inputs to the next, enhancing interoperability between subtasks.
    7.3. If a reasoning flaw is identified by expert evaluation, guide the agents' instructions to avoid repeating the same mistake.
    7.4. Use feedbacks from experts to guide agents's instructions.
8. If the evaluation is empty, treat this as the first attempt and focus on creating a clear, logical, and detailed decomposition.
9. Ensure the decomposition is tailored to the query’s intent and context, leveraging any available information from previous tasks or the query itself.
10. Consider output format of each subtask, so that they could pass their output to other agents effectively.

[Previous Task Decomposition]
{prev_task_decomposition}

Return in JSON format, contains 'task_decomposition' and 'thought. 
{{
    'thought': "Your thought while decomposing task to subtasks",
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

    task_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'task_decomposition'], 0.0)   
    
    return task_decomposition 

async def refined_workflow(meta_model, prev_workflow, interaction_pattern, evaluation_feedback, query):
    user_prompt_generate_workflow = f"""
You are tasked with instantiating a concrete version of an abstract multi-stage workflow to solve complex queries using agent-based reasoning techniques (e.g., Chain-of-Thought, Self-Consistency, Reflexion, Debate). The workflow must be customized for the provided query and incorporate feedback from previous attempts to improve contextual usage, reasoning processes, and overall effectiveness.

[Important Notes]
1. You may adapt the agent interaction patterns (e.g., CoT, Self-Consistency CoT, Debate, Reflexion) to best suit the query’s complexity and requirements.
2. The generated code must include sufficient subtasks as planned in the task decomposition.
3. A stage may include multiple steps if necessary to achieve the query’s objectives.
4. The final answer must be processed by `make_final_answer` function.
5. Subtask id must start from 1 to n, eg. 1, 2, 3, 4, 5, ...
6. Follow strictly the design of defined interaction patterns.
7. You can only use `self.cot`, `self.sc_cot`, `self.debate` and `self.reflexion`.

[Instructions]
You must follow the requirements below when generating the concrete workflow:

1. Return Format
Reply EXACTLY in the following JSON format. The generated code MUST be in function format (i.e., forward function) without comments.
{{
    "thought": "Your thought while generating code."
    "code": "Your code."
}}
You must remove commnets from generated code.
Ensure the response is a well-formed JSON object and includes all required fields.

2. Agent Collaboration Patterns
    3.1. Implement agent interaction structures such as Chain-of-Thought (CoT), Self-Consistency CoT (SC-CoT), Reflexion, or Debate, selecting the most appropriate pattern for each subtask based on its complexity and the query’s requirements.
    3.2. Retain variables like self.node_model, self.debate_role, self.max_sc, and self.max_round with their correct roles.
    3.3. Refer to the provided interaction pattern examples for implementation guidance. 
    3.4. You can only use `self.cot`, `self.sc_cot`, `self.debate` and `self.reflexion`.
[Interaction Pattern] {interaction_pattern}

3. Apply the feedback from previous reasoning process to enhance reasoning process of generated workflow.
Here are evaluations and suggestions from experts
{evaluation_feedback}

Ensure you refine previous workflow throughly.
[Previous Workflow]
{prev_workflow}

4. Placeholder Replacement
    4.1. Replace placeholders (e.g., [final answer], [condition #1]) with specific information relevant to the subtask. For example, if the query involves calculating the speed of a nucleus, replace [final answer] with “the speed of the nucleus.”
    4.2. Ensure reasoning goals are fully grounded in the query’s context for clarity and relevance.

[Your Task]
Generate a concrete agentic workflow in Python code format, based on the Abstract Workflow, tailored to the query and informed by the evaluation feedback.

[Workflow Template]
{ABSTRACTED_WORKFLOW_TEMPLATE}

[Query]
{query}
    """
    
    msg_list = [
        {"role": "user", "content": user_prompt_generate_workflow},
    ]

    next_solution ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'code'], 0.0)
    print("================ Generated Multi-agent system ================\n", next_solution['code'])
    next_solution['code'] = next_solution['code'].replace("{{", "{")
    next_solution['code'] = next_solution['code'].replace("}}", "}")
    
    return next_solution
 
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

    if example_id < 196:
        return 0, 0, ""

    start_time_ = time.time()
    
    global_node_model = get_global("global_node_model")
    
    # declare results path
    result_path = expr_name + f"{args.dataset}"
    expr_name = expr_name + f"{args.dataset}/{example_id}/{meta_model}_{args.node_model}_{verifier_model}"
    oracle_acc_result_path = f'results/{args.dataset}/abstract_workflow_dev_3_7/{meta_model}_{global_node_model}_oracle.results'
    oracle_acc_path = Path(oracle_acc_result_path)
    oracle_acc_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path = os.path.join(args.save_dir, f"{expr_name}_{args.option}_judge")
    os.makedirs(os.path.dirname(judge_path), exist_ok=True)
    global_judge_path = get_global('global_judge_path')
    global_judge_path[str(example_id)] = judge_path
    final_results_path = f'results/{args.dataset}/abstract_workflow_dev_3_7/{meta_model}_{global_node_model}/final_results_{example_id}.json'
    result_path = Path(final_results_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
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
        'gpt-4o_chatgpt',
        'gpt-4.1-mini'
    ]

    max_workers = 1
    set_global("global_max_workers", max_workers)
    
    # load mas chains list from offline abstract workflows
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/abstracted_workflow/workflow_chains.json', 'r', encoding='utf-8') as f:
        default_mas_chain = json.load(f)
        
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
    high_level_decomposition = await high_level_task_decomposition(meta_model, task_queue[0].content)
    print("high_level_decomposition: ", high_level_decomposition['subtask_list'])
    
    high_level_subtasks = [subtask['objective'] for subtask in high_level_decomposition['subtask_list']]
    
    # embeddings, postprocess and clustering the embeddings
    abstractor = MASAbstraction()
    abstracted_subtasks = await abstractor.abstract_task_decomposition(task_queue[0].content, high_level_subtasks)
    merged_subtasks = [f"{a_subtask['subtask_name']}: {a_subtask['abstracted_objective']}" for a_subtask in abstracted_subtasks]
    print(merged_subtasks)
    
    embeddings = await abstractor.embedding_subtask(merged_subtasks)
    
    # get kmeans clustering object from former abstraction
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/kmeans.pkl', 'rb') as f:
        kmeans = pkl.load(f)
    
    # get pca object from former abstraction, which fit with available data
    with open('workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v2/pca.pkl', 'rb') as f:
        pca = pkl.load(f)
        
    normalized_embeddings = normalize(embeddings, norm="l2")
    reduced_embeddings = pca.transform(normalized_embeddings)
    
    cluster_id = kmeans.predict(reduced_embeddings)
    mas_chain = []
    for idx in cluster_id:
        if str(idx) not in mas_chain:
            mas_chain.append(str(idx))

    # choose the most similar abtracted workflow, based on levenshtein distance
    sorted_chains = sorted(default_mas_chain, key=lambda mas: levenshtein_array_to_array(mas_chain, mas))
    closest_1 = sorted_chains[:1]

    workflow_index = 0
    max_levenshtein_distance = 0
    for idx, awd in enumerate(abstract_workflow):
        
        if levenshtein_array_to_array(awd['chain'], closest_1[0]) > max_levenshtein_distance:
            max_levenshtein_distance = levenshtein_array_to_array(awd['chain'], closest_1[0])
            workflow_index = idx
            
    workflow_index = [workflow_index]
    
    print("workflow index: ", workflow_index)
    filterd_workflow = [abstract_workflow[idx] for idx in workflow_index]
    print("Filtered workflow: ", [fw['name'] for fw in filterd_workflow])
    
    task_content = task_queue[0].content
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, task_content, task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
    
    global_task_queue = get_global('global_task_queue')
    global_task_queue[str(example_id)] = task_queue_tmp    
    set_global("global_task_queue", global_task_queue)
    
    aw = filterd_workflow[0]
    print(f'============Initial Archive: {aw["name"]}=================')
    default_global_n = get_global("global_n")
    default_global_n[str(example_id)] = f"{aw["name"]}_{example_id}"
    set_global("global_n", default_global_n)

    global_n = get_global("global_n")
    global_n = global_n[str(example_id)]
    global_ns.append(global_n)
    
    '''
    ================================================== CONCRETIZE WORKFLOW ======================================================
    '''
    
    # load chosen abstracted workflow
    with open(aw['code_path'], 'r', encoding='utf-8') as f:
        abstracted_mas = json.load(f)
    
    collaboration_dependencies = {subtask['subtask_id']: {'agent_collaboration': subtask['agent_collaboration'], 'dependencies': subtask['dependencies']} for subtask in abstracted_mas}
    
    # decompose query into multiple subtasks
    task_decomposition = await specific_task_decomposition(meta_model, task_queue_tmp[0].content, output_description, aw['flow'], collaboration_dependencies)
    print("\n============= Task Decomposition: =============\n", task_decomposition['task_decomposition'])
    stage_desc = str(aw).replace("subtask", "stage")
    
    # generate new workflow that concretized for this query
    next_solution = await generate_concretized_workflow(meta_model, task_decomposition['task_decomposition'], output_description, interaction_pattern, stage_desc, task_queue_tmp[0])
    print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
    
    # evaluate multi agent system
    try:
        next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
        acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"])
    except Exception as e:
        print("Error: ", str(e))
        error_trace = traceback.format_exc()
        print("Full error trace:\n", error_trace)
        return -1, -1, ""
    
    # with open("logs.pkl", "wb") as f:
    #     pkl.dump(logs, f)
    
    # save results
    judge_path = os.path.join(args.save_dir, f"{expr_name}_{aw['name']}_{example_id}_full_response")
    with open(judge_path, 'w') as judge_file:
        judge_file.write(f'Question: {task_queue[0].content}\nIteration: {aw['name']}\nFull Response:{raw_results}')

    if global_use_oracle_verifier:
        acc_list = acc_oracle_verifier_list
    else:
        acc_list = acc_model_verifier_list

    print(f"acc_list:", acc_list)
    print(f"mean acc_list:", np.mean(acc_list))
    
    extracted_answer = re.search(ANSWER_PATTERN, final_reponse[0]).group(1)     

    if '[TOO_HARD]' in extracted_answer:
        extracted_answer = extracted_answer[:extracted_answer.index('[TOO_HARD]')]        

    converted_code_filename = os.path.join(args.save_dir, f'{expr_name}_{aw["name"]}_{example_id}_first_gen_converted.py')
    with open(converted_code_filename, "w") as fh:
        fh.write(next_solution['code'])    
    
    print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))

    with open(oracle_acc_result_path, "a+") as fh:
        fh.write(f'experiemnt {example_id}: 1 (initial {aw["name"]}): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')
    
    max_score = max(max_score, acc_oracle_verifier_list[0])
    final_results.append({
        "example_id": example_id,
        "score": max_score,
        "max_cost": get_global("global_COST_TOTAL")
    })
    
    '''
    ================================================== REFINE WORKFLOW ======================================================
    '''
    
    # if the first concretized workflow is not optimal, try to evaluate and refine it
    if max_score == 0:
        subtask_desc = logs
            
        # get evaluation from many llm experts
        evaluation = await evaluate_workflow(task_queue_tmp[0].content, subtask_desc, current_ans, output_description, verifier_hub)
        synthesized_evaluation = await synthesize_evaluations(verifier_model, evaluation, task_queue_tmp[0].content)
        
        # regenerate task decomposition
        # regenerated_task_decomposition = await refined_task_decomposition(meta_model, task_queue_tmp[0].content, output_description, synthesized_evaluation, task_decomposition['task_decomposition'])
        # print(regenerated_task_decomposition["task_decomposition"])
        
        # regenerate workflow
        next_solution = await refined_workflow(meta_model, next_solution['code'], interaction_pattern, synthesized_evaluation, task_queue_tmp[0])
        print("Total cost in the loop: ", get_global("global_COST_TOTAL"))
        
        # re-evaluate refined workflow
        try:
            next_solution['code'] = next_solution['code'].replace("forward", f"forward_{example_id}")
            acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, logs, current_ans, ground_truth, total_time = await evaluate_forward_fn(args, example_id, next_solution["code"])
        except Exception as e:
            print("Error: ", str(e))
            error_trace = traceback.format_exc()
            print("Full error trace:\n", error_trace)
        
        # save refined results
        judge_path = os.path.join(args.save_dir, f"{expr_name}_{aw['name']}_{example_id}_full_response_refined")
        with open(judge_path, 'w') as judge_file:
            judge_file.write(f'Question: {task_queue[0].content}\nIteration: {aw['name']}\nFull Response:{raw_results}')

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

        converted_code_filename = os.path.join(args.save_dir, f'{expr_name}_{aw["name"]}_{example_id}_converted_refined.py')
        with open(converted_code_filename, "w") as fh:
            fh.write(next_solution['code'])    

        print(f"COST_TOTAL:", get_global("global_COST_TOTAL"))

        with open(oracle_acc_result_path, "a+") as fh:
            fh.write(f'experiemnt {example_id}: 1 (initial {aw["name"]}): acc_oracle_verifier_list: {acc_oracle_verifier_list} acc_model_verifier_list: {acc_model_verifier_list}\n')
        
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
        "max_cost": get_global("global_COST_TOTAL")
    })
    
    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)
            
    return max_score, total_time, ""
        
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