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

client = openai.OpenAI()

Info = namedtuple('Info', ['name', 'author', 'content', 'prompt', 'sub_tasks', 'agents', 'iteration_idx'])

ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

async def run_code(code, entry_point):
    try:
        # Create a new global namespace
        global_namespace = {}
        # print("Create a new global namespace")
        disallowed_imports = [
            "os",
            "sys",
            "subprocess",
            "multiprocessing",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot",
            "pylab",
            "tkinter",
            "PyQt5",
            "wx",
            "pyglet",
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if entry_point in global_namespace and callable(global_namespace[entry_point]):
            result = global_namespace[entry_point]()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"

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
    
    async def cot(self, subtask_id, cot_agent_desc):
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=cot_agent_desc.get('temperature', 0.0))
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_agent_desc.get('instruction', "Please do it step-by-step."),
            "context": cot_agent_desc.get('context', ['user query']),
            "agent_collaboration": "CoT"
        }
        thinking, answer = await cot_agent(cot_agent_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'cot_agent': cot_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }, subtask_desc
    
    async def sc_cot(self, subtask_id, cot_agent_desc, n_repeat = 3):
        cot_agents = [LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=cot_agent_desc.get('temperature', 0.5)) for _ in range(n_repeat)]
        possible_answers = []
        list_thinking = []
        list_answer = []
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_agent_desc.get('instruction', "Please do step-by-step"),
            "context": cot_agent_desc.get('context', ['user query']),
            "agent_collaboration": "SC_CoT"
        }
        
        final_decision_desc = {
            'instruction': f"Make final answer of: {cot_agent_desc["instruction"]}. Make sure this answer is the most consistent and exact.",
            'temperature': 0.0
        }
        
        for i in range(n_repeat):
            # Each CoT-SC agent tries to calculate all possible cases independently
            thinking, answer = await cot_agents[i](cot_agent_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
            possible_answers.append(str(answer.content))
            list_thinking.append(thinking)
            list_answer.append(answer)
            
        # The most common answer is chosen for consistency and accuracy.
        final_decision_agent = LLMAgentBase(['thinking', 'answer'], "Final Decision Agent", 
                                            model=self.node_model, temperature=final_decision_desc.get('temperature', 0.0))
        thinking, answer = await final_decision_agent(cot_agent_desc.get('input', []) + list_thinking + list_answer, 
                                                    final_decision_desc.get('instruction', "Please do step-by-step"), 
                                                    is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'cot_agent': cot_agents,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc,
            # 'list_thinking': list_thinking,
            # 'list_answer': list_answer
        }, subtask_desc
    
    async def reflexion(self, subtask_id, reflect_desc, n_repeat = 1):
        reflect_inst =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct"
        
        critic_desc = {
            'temperature': 0,
            "instruction": "Review potential solutions and analyze their limitations."
        }
        
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", model=self.node_model, temperature=reflect_desc.get('temperature', 0.0))
        critic_agent = LLMAgentBase(['feedback', 'correct'], "Critic Agent", model=self.node_model, temperature=critic_desc.get('temperature', 0.0))
        
        # Input for CoT agent
        cot_inputs = reflect_desc.get('input', [])
        
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": reflect_desc.get('instruction', "Please do step-by-step") + reflect_inst,
            "context": reflect_desc.get('context', ['user query']),
            "agent_collaboration": "Reflexion"
        }
        
        feedbacks, corrects = [], []
        thinkings, answers = [], []
        
        # Generate the first version
        thinking, answer = await cot_agent(cot_inputs, reflect_desc.get('instruction', "Please do step-by-step") + reflect_inst, 0, is_sub_task=True)
        thinkings.append(thinking)
        answers.append(answer)
        for i in range(n_repeat):
            # Critic agent debates and criticizes pros and cons of previous version
            feedback, correct = await critic_agent(reflect_desc.get('input', []) + [thinking, answer], 
                                        critic_desc.get('instruction', "Review the previous") + critic_inst, i, is_sub_task=True)
            feedbacks.append(feedback)
            corrects.append(correct)
            if correct.content == "True":
                break
            
            cot_inputs.extend([thinking, answer, feedback])
            thinking, answer = await cot_agent(cot_inputs, reflect_desc.get('instruction', "Please do step-by-step") + reflect_inst, i + 1, is_sub_task=True)
            thinkings.append(thinking)
            answers.append(answer)
            
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'cot_agent': cot_agent,
            'critic_agent': critic_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc,
            # 'list_thinking': thinkings,
            # 'list_answer': answers,
            # 'list_feedback': feedbacks,
            # 'list_correct': corrects
        }, subtask_desc
    
    async def debate(self, subtask_id, debate_desc, n_repeat = 1):
        debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        debate_agents = [LLMAgentBase(['thinking', 'answer'], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=debate_desc.get('temperature', 0.5)) 
                        for role in self.debate_role]

        all_thinking = [[] for _ in range(n_repeat)]
        all_answer = [[] for _ in range(n_repeat)]
        
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": debate_desc.get('instruction', "Please do step-by-step") + debate_instr,
            "context": debate_desc.get('context', ['user query']),
            "agent_collaboration": "Debate"
        }
        
        final_decision_desc = {
            'instruction': f"Make final answer of : {debate_desc["instruction"]}",
            'temperature': 0.0
        }
        
        for r in range(n_repeat):
            # N_max_5 rounds of debating
            for i, agent in enumerate(debate_agents):
                # Each agent proposes its solution
                if r == 0:
                    thinking, answer = await agent(debate_desc.get('input', []), 
                                            debate_desc.get('instruction', "Please do step-by-step") + debate_instr, r, is_sub_task=True)
                else:
                    # Generate next solution based on comments and counter-arguments from other debaters
                    input_infos = debate_desc.get('input', []) + all_thinking[r-1] + all_answer[r-1]
                    thinking, answer = await agent(input_infos, debate_desc.get('instruction', "Please do step-by-step") + debate_instr, r, is_sub_task=True)
                
                all_thinking[r].append(thinking)
                all_answer[r].append(answer)
        
        # Final decision agent makes final decision
        final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
        final_decision_agent = LLMAgentBase(['thinking', 'answer'], "Final Decision Agent", 
                                            model=self.node_model, temperature=final_decision_desc.get('temperature', 0.0))
        thinking, answer = await final_decision_agent(debate_desc.get('input', []) + all_thinking[-1] + all_answer[-1], 
                                                    final_decision_desc.get('instruction', "Please do step-by-step") + final_instr, 
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
            # 'list_thinking': all_thinking,
            # 'list_answer': all_answer,
        }, subtask_desc
        
    async def answer_generate(self, subtask_id, cot_agent_desc):
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Answer Generate Agent", model=self.node_model, temperature=cot_agent_desc.get('temperature', 0.0))
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_agent_desc.get('instruction', "Please do step-by-step") + "\nThink step by step and solve the problem. Ensure the final answer is concisely and clearly, direct response to the question without including explanations or reasoning",
            "context": cot_agent_desc.get('context', ['user query']),
            "agent_collaboration": "AnswerGenerate"
        }
        
        thinking, answer = await cot_agent(cot_agent_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'cot_agent': cot_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }, subtask_desc
        
    async def specific_format(self, subtask_id, formatter_desc):
        formatter_agent = LLMAgentBase(['thinking', 'answer'], "SpecificFormatter Agent", model=self.node_model, temperature=formatter_desc.get('temperature', 0.0))
       
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": formatter_desc.get('instruction', "Please do step-by-step") + f"\nExtract the correct answer for the question from the previous context. Then, return the final answer in the following format: {formatter_desc.get('format', "Provide short and concise answer, without explaination or additional information")}",
            "context": formatter_desc.get('context', ['user query']),
            "agent_collaboration": "SpecificFormat"
        }
        
        thinking, answer = await formatter_agent(formatter_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'formatter_agent': formatter_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }, subtask_desc
        
    async def aggregate(self, subtask_id, aggregate_desc):
        print(self.node_model)
        aggregate_agent = LLMAgentBase(['thinking', 'answer'], "Aggregate Agent", model=self.node_model, temperature=aggregate_desc.get('temperature', 0.0))
       
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": aggregate_desc.get('instruction', "Please do step-by-step") + f"\nCarefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution",
            "context": aggregate_desc.get('context', ['user query']),
            "agent_collaboration": "AggregateAgent"
        }
        
        thinking, answer = await aggregate_agent(aggregate_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'aggregate_agent': aggregate_agent,
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }, subtask_desc
        
    async def code_generate(self, subtask_id, code_generate_desc):
    
        code_generate_agent = LLMAgentBase(['thinking', 'code'], "Code Generate Agent", model=self.node_model, temperature=code_generate_desc.get('temperature', 0.0))
       
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": code_generate_desc.get('instruction', "Please do step-by-step") + f"""
\nWrite complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.
The entry point of the function: {code_generate_desc.get('entry_point', "solve")}.
Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
                """,
            "context": code_generate_desc.get('context', ['user query']),
            "agent_collaboration": "CodeGenerate"
        }
        
        thinking, code = await code_generate_agent(code_generate_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = { 
            "thinking": thinking,
            "code": code
        }
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}")
        print(f"Code: {str(code.content)}")
        
        return {
            'code_generate_agent': code_generate_agent,
            'thinking': thinking,
            'answer': code,
            'subtask_desc': subtask_desc
        }, subtask_desc
    
    async def exec_code(self, code, entry_point, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()

        try:
            result = await asyncio.wait_for(run_code(code, entry_point), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return "Error", "Code execution timed out"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"
        
    async def programmer(self, subtask_id, programmer_desc):

        code = None
        output = None
        feedback = ""
        
        for i in range(2):
            results = await self.code_generate(subtask_id, programmer_desc)
            code = results.get("answer")
            if isinstance(code, Info):
                code = code.content 
            if not code:
                
                results['subtask_desc']['exec_status'] = "Error"
                results['subtask_desc']['exec_result'] = "No code generated"
                results['answer'].content = "No code generated"
                results['answer'] = Info(results['answer'].name, results['answer'].author, "No code generated", results['answer'].prompt, results['answer'].sub_tasks, results['answer'].agents, results['answer'].iteration_idx)
                print(f"Subtask {subtask_id}, generated code: {str(code)}, status: No code generated")
                
                return {
                    "programmer_agent": results['code_generate_agent'], 
                    "thinking": results['thinking'],
                    "answer": results['answer'], 
                    "code": code,
                    "exec_status": "Error",
                    "exec_result": "No code generated",
                    "subtask_desc": results['subtask_desc']
                }, subtask_desc
            print("Code: ", code)
            status, output = await self.exec_code(code, programmer_desc.get('entry_point', "solve"))
            if status == "Success":
                results['subtask_desc']['exec_status'] = status
                results['subtask_desc']['exec_result'] = output
                results['answer'] = Info(results['answer'].name, results['answer'].author, output, results['answer'].prompt, results['answer'].sub_tasks, results['answer'].agents, results['answer'].iteration_idx)
                print(f"Subtask {subtask_id}, generated code: {str(code)}, status: {str(status)}, output: {str(output)}")
                
                return {
                    "programmer_agent": results['code_generate_agent'], 
                    "thinking": results['thinking'],
                    "answer": results['answer'], 
                    "code": code,
                    "exec_status": status,
                    "exec_result": output,
                    "subtask_desc": results['subtask_desc']
                }, subtask_desc
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
                
                programmer_desc['instruction'] = programmer_desc.get('instruction', "Please do step-by-step") + feedback

            # Force garbage collection after each iteration
            import gc
            gc.collect()
            
            results['subtask_desc']['exec_status'] = status
            results['subtask_desc']['exec_result'] = output
            results['answer'] = Info(results['answer'].name, results['answer'].author, output, results['answer'].prompt, results['answer'].sub_tasks, results['answer'].agents, results['answer'].iteration_idx)

        print(f"Subtask {subtask_id}, generated code: {str(code)}, status: {str(status)}, output: {str(output)}")

        return {
            "programmer_agent": results['code_generate_agent'], 
            "thinking": results['thinking'],
            "answer": results['answer'], 
            "code": code,
            "exec_status": status,
            "exec_result": output,
            "subtask_desc": results['subtask_desc']
        }, subtask_desc
    
    async def review(self, subtask_id, review_desc):
        review_agent = LLMAgentBase(['thinking', 'answer'], "Revise Agent", model=self.node_model, temperature=review_desc.get('temperature', 0.0))
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": review_desc.get('instruction', "Please do step-by-step") + "\nProvided solution which is just reviewed as incorrect, your task is to revise the solution to solve the question. Ensure the revised solutions is clear and correct",
            "context": review_desc.get('context', ['user query']),
            "agent_collaboration": "Review"
        }
        
        thinking, answer = await review_agent(review_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'review_agent': review_agent, 
            'thinking': thinking,
            'answer': answer,
            'subtask_desc': subtask_desc
        }, subtask_desc
        
    async def revise(self, subtask_id, revise_desc):
        revise_agent = LLMAgentBase(['thinking', 'revised_solution'], "Revise Agent", model=self.node_model, temperature=revise_desc.get('temperature', 0.0))
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": revise_desc.get('instruction', "Please do step-by-step") + "\nProvided solution which is just reviewed as incorrect, your task is to revise the solution to solve the question. Ensure the revised solutions is clear and correct",
            "context": revise_desc.get('context', ['user query']),
            "agent_collaboration": "Revise"
        }
        
        thinking, revised_solution = await revise_agent(revise_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
        subtask_desc['response'] = {
            "thinking": thinking,
            "revised_solution": revised_solution
        }
        
        print(f"Subtask {subtask_id}, thinking: {str(thinking.content)}, answer: {str(answer.content)}")
        
        return {
            'revise_agent': revise_agent,
            'thinking': thinking,
            'revised_solution': revised_solution,
            'subtask_desc': subtask_desc
        }, subtask_desc

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

async def high_level_task_decomposition(meta_model, query, n_repeat):
    potential_high_level_plan = []
    
    for i in range(0, n_repeat):
        task_high_level_decomposition = f"""
Role and Responsibility:
You are an expert LLM assistant trained to analyze and decompose a user query into its core subtasks. Your goal is to:

- Identify and list essential subtasks that are necessary to accomplish the user’s input query.
- Each subtask must represent a distinct and meaningful action or process step that contributes directly to solving the overall task, do not analyze too specific to each details in query.
- These subtasks must follow the reasoning model, from subtasks 1 -> subtask 2.
- Remove many trivial steps that are not the core subtasks.
- Maximum 4 subtasks.
User query:
{query}

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

        high_level_decomposition ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['subtask_list'], 0.0)
        print(f"\n=========== high level decomposition, attempt {i} ==========\n", high_level_decomposition)
        potential_high_level_plan.append(high_level_decomposition['subtask_list'])
        
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
    
    return final_high_level_decomposition['subtask_list']

async def task_analysis(meta_model, query):
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
{query}

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
    
    return detailed_analysis['analysis']

async def specific_task_decomposition(meta_model, query, collaboration_dependencies, aw_flow = None, prev_task_decomposition = None, evaluation = None, is_refinement = False):
    # processed_aw_flow = [v for k, v in aw_flow.items()]
    
    if not is_refinement:
        user_prompt = f"""
You are an agent specializing in task decomposition. Given a query, your job is to break it into subtasks with clear objectives and dependencies. 

[Inputs]
- Query: {query}
- Abstract Workflow: {aw_flow}
- Dependencies and Agent Collaboration: {collaboration_dependencies}

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
- Query: {query}
- Evaluation from previous attempts: {evaluation}
- Previous decomposition: {prev_task_decomposition}
- Dependencies and Agent Collaboration: {collaboration_dependencies}

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
            
    return task_decomposition['task_decomposition']
    
async def generate_concretized_workflow(meta_model, task_decomposition, interaction_pattern, query):
    user_prompt_generate_workflow = f"""
You are tasked with generating a concrete multi-stage agentic workflow using techniques like Chain-of-Thought, Self-Consistency, Reflexion, or Debate. Your goal is to implement the workflow for the provided query, following the task decomposition and improving from previous feedback.

Ensure this code is runnable. Double-check many times.

[Inputs]
1. Query: {query}
2. Task Decomposition: {task_decomposition}
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
    
    return next_solution
    
async def evaluate_workflow(query, subtask_desc, current_ans, output_description, verifier_hub):
    reasoning_refinement_prompt = f"""
You are a Verification Agent. Your task is to review the reasoning process of previous agents to find errors, check if contexts were sufficient, analyze how agents interacted, and suggest workflow improvements. The final answer is known to be wrong based on professor feedback — identify why and how to fix it.

# Input
1. User Query: {query}
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
    
    tasks = [
        (verifier_model, asyncio.create_task(
            get_json_response_from_gpt(copy.deepcopy(msg_list), verifier_model, ['suggestion', 'feedback', 'failure_reason'], 0.0)
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
            "verifier_name": verifier_model,
            "evaluation": mas_feedback
        })
        
    for ev in evaluation:
        print(f"\n================= {ev['verifier_name']} ==================\n")
        print(ev["evaluation"])
    
    return evaluation

async def merge_filtered_workflow(meta_model, filtered_sequential_only_workflow, filtered_loop_contain_workflow, filtered_conditional_contain_workflow):
    user_prompt_merge_workflow = f"""
You are a specialized AI agent for analyzing and merging task workflows represented in structured JSON format. Your task is to intelligently merge two similar but not identical workflows into a unified, optimized version that preserves both shared and unique subtasks, while maintaining logical structure and relative control flow positions.

### 🔧 INPUT

You will be given **two workflows**, each represented as a list of subtasks in JSON format. Each subtask has the following fields:
- `subtask_id`: unique identifier
- `subtask_name`: human-readable name
- `abstracted_objective`: a concise description of what the subtask accomplishes
- `agent_collaboration`: either a string (e.g., "logic code") or a list of collaboration methods (e.g., `["CoT", "AnswerGenerate"]`)
- `dependencies` (optional): list of subtask_ids this subtask depends on

Each workflow may also contain **control flow markers**:
- `start_sequential` / `end_sequential`
- `start_loop` / `end_loop`
- `start_conditional` / `end_conditional`
- `start_true_branch` / `end_true_branch`
- `start_false_branch` / `end_false_branchs`

### 📌 TASK

1. **Merge the two workflows** into a single workflow list that:
   - Keeps the relative position of control flows (`start_sequential`, `start_loop`, etc.) **exactly as they appear** in each respective flow.
   - If two subtasks serve **identical purposes** and appear at similar relative positions, **merge them into one**.
     - Consider subtasks as equivalent if their `abstracted_objective` and `agent_collaboration` are very similar and they appear in the same structural place (e.g., inside a loop).
   - Keep **unique subtasks** from each workflow, ensuring there is **no redundant repetition**.
   - Adjust `dependencies` accordingly in the merged version.
   - If a subtask is present in only one workflow but logically fits into the merged structure, include it and annotate it with `"optional": true`.

2. **Ensure logical consistency**: 
   - Do not place subtasks outside of their valid control flow (e.g., don't put a looping subtask outside a loop).
   - Do not duplicate identical or redundant subtasks.
   - Maintain correct ordering of dependencies.

3. **Avoid**:
   - Breaking control flow structure (e.g., nesting start/end incorrectly).
   - Losing important agent collaboration roles or objectives.
   - Blindly appending all subtasks together.

### 🧾 OUTPUT FORMAT

Return results in JSON format, which include `thought` (Describe your reasoning for decomposing tasks, interpreting control flows, choosing agent strategies, and sequencing subtasks.)
and `merge_filtered_workflow` that represents the merged workflow.
In merge_filtered_workflow, it could contain `Control Flow` (start_sequential, end_sequential, ....) or `Stage`. Each stage include multiple subtasks. Each of them contains:
- `subtask_id`
- `subtask_name`
- `abstracted_objective`
- `agent_collaboration`
- `dependencies` (optional)

You will be provided with:
- Filtered Sequential Only Workflow:
{filtered_sequential_only_workflow}

- Filtered Loop Contain Workflow:
{filtered_loop_contain_workflow}

- Filtered Conditional Contain Workflow:
{filtered_conditional_contain_workflow}

### 🧠 Reasoning Strategy

While merging, follow these reasoning steps:
- Align control flow structure first.
- Identify matching subtasks by comparing their names, objectives, collaboration patterns, and structural position.
- Merge identical subtasks, keep unique ones if relevant.

Example:
{{
    "thought": "Describe your reasoning for decomposing tasks, interpreting control flows, choosing agent strategies, and sequencing subtasks.",
    "merge_filtered_workflow": {{
        "Control Flow 0": {{
            "flow_type": "start sequential",
            "flow_desc": "..."
        }},
        "Stage 0": {{
            "subtask_1": {{
                "objective": "...",
                "agent_collaboration": "CoT | SC_CoT | Debate | Reflexion | AnswerGenerate | SpecificFormat | Aggregate | CodeGenerator | Programmer | Review | Revise",
                "dependencies": []
            }},
            "subtask_2": {{
                "objective": "...",
                "agent_collaboration": "...",
                "dependencies": ["subtask_1"]
            }}
        }},
        .....
        "Control Flow n": {{
            "flow_type": "end sequential",
            "flow_desc": "..."
        }}
    }}
}}
    """
    
    msg_list = [
        {"role": "user", "content": user_prompt_merge_workflow},
    ]

    merge_filtered_workflow ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), meta_model, ['thought', 'merge_filtered_workflow'], 0.0)
    print("================ Merged Filtered Workflow ================\n", merge_filtered_workflow['merge_filtered_workflow'])

    return merge_filtered_workflow['merge_filtered_workflow']
 
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
            
async def apply_abstract_workflow_enhance(args, expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow = None, date_time=""):

    if example_id < 150:
        return -1, -1, -1, ""

    start_time_ = time.time()
    total_execution_time = 0
    
    global_node_model = get_global("global_node_model")
    save_file = "dev_22"
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
    high_level_decomposition = await high_level_task_decomposition(meta_model, task_queue[0].content, 5)
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
    for idx in cluster_id:
        if str(idx) not in mas_chain:
            mas_chain.append(str(idx))

    # remove start and end step of control flow from mas
    
    # filter sequential only mas
    sequential_only_mas = []
    loop_contain_mas = []
    conditional_contain_mas = []
    
    for idx, mas in enumerate(default_mas_chain):
        contain_loop = False
        contain_conditional = False
        # if 'start_conditional' in mas:
        #     conditional_contain_mas.append({
        #         'id': idx,
        #         'mas': mas
        #     })
        #     contain_conditional = True
        
        # if 'start_loop' in mas:
        #     loop_contain_mas.append({
        #         'id': idx,
        #         'mas': mas
        #     })
        #     contain_loop = True
        
        if not contain_conditional and not contain_loop:
            sequential_only_mas.append({
                'id': idx,
                'mas': mas
            })
            
    async def choose_the_most_similar_mas(mas_chain, mas_chain_list, abstract_workflow_):
        if len(mas_chain_list) == 0:
            return None
        
        mas_chain_no_flow = []
        for mas in mas_chain_list:
            mas_no_flow = []
            for step in mas['mas']:
                if isinstance(step, str) and (step.startswith("start") or step.startswith("end")):
                    continue
                mas_no_flow.append(step)
                
            mas_chain_no_flow.append({
                'id': mas['id'],
                'mas': mas_no_flow
            })
        
        # choose the most similar abtracted workflow, based on levenshtein distance  
        sorted_chains = sorted(mas_chain_no_flow, key=lambda mas: levenshtein_array_to_array(mas_chain, mas['mas']))
        # print(sorted_chains)
        closest_1 = sorted_chains[:1]

        workflow_index = 0
        max_levenshtein_distance = 0
        for idx, awd in enumerate(abstract_workflow_):
            if levenshtein_array_to_array(awd['chain'], closest_1[0]['mas']) > max_levenshtein_distance:
                max_levenshtein_distance = levenshtein_array_to_array(awd['chain'], closest_1[0]['mas'])
                workflow_index = closest_1[0]['id']
                
        workflow_index = [workflow_index]
        
        print("workflow index: ", workflow_index)
        filterd_workflow = [abstract_workflow_[idx] for idx in workflow_index]
        print("Filtered workflow: ", [fw['name'] for fw in filterd_workflow])
        
        return filterd_workflow

    # choose the most similar sequential only mas
    filtered_sequential_only_workflow = await choose_the_most_similar_mas(mas_chain, sequential_only_mas, abstract_workflow)
    print("Query-based chain: ", mas_chain)
    print("Similar chain: ", filtered_sequential_only_workflow[0]['chain'])
    print("Levenshtein distance: ", levenshtein_array_to_array(mas_chain, filtered_sequential_only_workflow[0]['chain']))
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "Abstract MAS choosing"
        content = f"Query-based chain: {mas_chain}\nSimilar chain: {filtered_sequential_only_workflow[0]['chain']}\nLevenshtein distance: {levenshtein_array_to_array(mas_chain, filtered_sequential_only_workflow[0]['chain'])}"
        
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
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
    
    task_detail_analysis = await task_analysis(meta_model, task_queue[0].content)
    print(f"\n================== Query analysis: {task_detail_analysis}========================\n")
    task_queue_tmp = [Info(task_queue[0].name, task_queue[0].author, str(task_content) + "\n\nDetailed Analysis: \n" + str(task_detail_analysis), task_queue[0].prompt, task_queue[0].sub_tasks, task_queue[0].agents, task_queue[0].iteration_idx)]
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
    if filtered_sequential_only_workflow:
        with open(filtered_sequential_only_workflow[0]['code_path'], 'r', encoding='utf-8') as f:
            sequential_only_aw_ = json.load(f)
        sequential_only_aw = {
            'flow': filtered_sequential_only_workflow[0]['flow'],
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
    
    # decompose query into multiple subtasks
    task_decomposition = await specific_task_decomposition(meta_model, task_queue_tmp[0].content, dependencies_and_agent_collaboration, sequential_only_aw['flow'])
    print("\n============= Task Decomposition: =============\n", task_decomposition)
    with open(log_path, "a+", encoding="utf-8") as f:
        
        phase = "Task Decomposition"
        content = task_decomposition
        f.write(f"\n============== {phase} ================\n")
        f.write(str(content))
    # return 1, 1, 1, ""
    stage_desc = str(sequential_only_aw['flow']).replace("subtask", "stage")
    
    # generate new workflow that concretized for this query
    next_solution = await generate_concretized_workflow(meta_model, task_decomposition, interaction_pattern, task_queue_tmp[0].content)
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
        evaluation = await evaluate_workflow(task_queue_tmp[0].content, subtask_desc, current_ans, output_description, verifier_hub)
        with open(log_path, "a+", encoding="utf-8") as f:
        
            phase = "Evaluation from verifiers"
            content = evaluation
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))

        refined_task_decomposition = await specific_task_decomposition(meta_model, task_queue_tmp[0].content, dependencies_and_agent_collaboration, evaluation=evaluation, prev_task_decomposition=task_decomposition, is_refinement=True)
        
        print("\n============= Refined Task Decomposition: =============\n", refined_task_decomposition)
        with open(log_path, "a+", encoding="utf-8") as f:
            
            phase = "Refined Task Decomposition"
            content = refined_task_decomposition
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
        # regenerate workflow
        next_solution = await generate_concretized_workflow(meta_model, refined_task_decomposition, interaction_pattern, task_queue_tmp[0].content)
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
        
async def test_operator(args, expr_name, example_id, task_queue, meta_model, verifier_model, pattern = None):

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
    result_path = f'results/{args.dataset}/single_agent_baselines_v3/{pattern}/{meta_model}_{global_node_model}_{verifier_model}.results'
    oracle_acc_result_path = f'results/{args.dataset}/single_agent_baselines_v3/{pattern}/{meta_model}_{global_node_model}_oracle.results'
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
    
    final_results_path = f'results/{args.dataset}/single_agent_baselines_v3/{pattern}/{meta_model}_{global_node_model}/final_results_{example_id}.json'
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
        workflow["code"] = '''
async def forward(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []
    logs =  []
    
    programmer_instruction1 = "Sub-task 1: Generate Python runnable code that addresses the following problem: [problem1]"
    programmer_desc1 = {
        'instruction': programmer_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query"],
        'entry_point': "solve"
    }
    results1 = await self.programmer(
        subtask_id="subtask_1", 
        programmer_desc=programmer_desc1
    )
    
    agents.append(f"Programmer Agent {results1['programmer_agent'].id}, generate code for problem [problem #1], thinking: {results1['thinking'].content}; answer: {results1['answer'].content}, executing reults: {results1['exec_result']}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}; output - {results1['exec_result']}")
    logs.append(results1['subtask_desc'])
    
    final_answer = await self.make_final_answer(results1['thinking'], results1['answer'], sub_tasks, agents)
    return final_answer, logs
'''
        acc_oracle_verifier_list, acc_model_verifier_list, results, _, _, final_reponse, raw_results, _, _, _, total_time = await evaluate_forward_fn(args, example_id, workflow["code"])
    except Exception as e:
        print("Error: ", str(e))
        error_trace = traceback.format_exc()
        print("Full error trace:\n", error_trace)
    
    return 0, 0, ""