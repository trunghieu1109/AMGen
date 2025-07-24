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
import asyncio

Info = namedtuple('Info', ['name', 'author', 'content', 'prompt', 'sub_tasks', 'agents', 'iteration_idx', 'test', 'entrypoint'])

ROLE_DESC = lambda role: f"You are a {role}."
SYSTEM_MSG = ""

PRINT_LLM_DEBUG = False
SEARCHING_MODE = True

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
                (field_name, author, content, prompt, _, _, iteration_idx, _, _) = input_info
            else:
                continue
            if author == await self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                if is_sub_task: 
                    input_infos_text += f' Related original question:\n\n{content}\n\nThese are just pieces of information related to the question. You are not required to answer the question — just follow what is defined in the instruction: {instruction}.   \n\nThese are many questions and answers that implemented before current subtasks:\n\n'
                else:
                    input_infos_text += f'{content}\n\n'
            elif iteration_idx != -1:
                if is_sub_task and prompt is not None: 
                    extracted_question =  await self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'###Question: {extracted_question} \n\n ### {field_name} in interation: {iteration_idx + 1} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question
                    else:
                        input_infos_text += f'###{field_name} in interation: {iteration_idx + 1} by {author}:\n{content}\n\n'

                else:
                    input_infos_text += f'###{field_name} in interation: {iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                if is_sub_task and prompt is not None: 
                    extracted_question = await self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'###Question: {extracted_question} \n\n ### {field_name} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question # we do not want to duplicate the prompt
                    else:
                        input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        if is_sub_task: 


            if global_format_choice == 'json':
                prompt = input_infos_text + f'''Given the above, answer the following question: {instruction}\n\n then justify completely and detailedly, step-by-step why you think so in the "thinking" entry. 
Consider all cases that are possible to happen.
Avoid some unclear explainations, such as "Using the circle equation or the condition for four points to be concyclic, we derive an equation in x. Solving this quadratic equation yields x=36.".
In default, return response in json format.
                '''
                
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
            
        with open("system_prompt.txt", "a+", encoding="utf-8") as f:
        
            phase = "system prompt"
            content = system_prompt
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
            
        with open("prompt1.txt", "a+", encoding="utf-8") as f:
        
            phase = "user prompt"
            content = prompt
            f.write(f"\n============== {phase} ================\n")
            f.write(str(content))
            
        return system_prompt, prompt

    async def query(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False) -> dict:
        

        system_prompt, prompt = await self.generate_prompt(input_infos, instruction, is_sub_task=is_sub_task)

        prompt = [
            await _pack_message(content=system_prompt, role="system"),
            await _pack_message(content=prompt, role="user")]
        # use system prompt
        # print("Execution model: ", self.model)
        response_json ,_ = await get_json_response_from_gpt(prompt, self.model, self.output_fields, self.temperature, is_execution=True)

        output_infos = []
        cnt = 0
        for key, value in response_json.items():
            info = Info(key, await self.__repr__(), str(value), prompt, None, None, iteration_idx, None, None)
            output_infos.append(info)
            cnt += 1
            if cnt == len(self.output_fields):
                break
        return output_infos

    async def __repr__(self):
        return f"{self.agent_name} {self.id}"

    async def __call__(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False):
        return await self.query(input_infos, instruction, iteration_idx=iteration_idx,  is_sub_task=is_sub_task)