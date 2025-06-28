import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import re
from typing import Any
from datasets import load_dataset
from utils import random_id
from shared_vars import set_global, get_global
from common import _pack_message, get_json_response_from_gpt, get_json_response_from_gpt_reflect
from sampler.chat_completion_sampler import ChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import ChatCompletionSampler as VllmChatCompletionSampler
import pandas as pd
from main_question import DataScorer


Info = namedtuple('Info', ['name', 'author', 'content', 'prompt', 'sub_tasks', 'agents', 'iteration_idx'])

ROLE_DESC = lambda role: f"You are a {role}."

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
        
    def extract_pattern(self, prompt):
        # pattern = r"\s*(.*?)\s*\n\nRelated original question"
        pattern = r"Given the above, answer the following question: \s*(.*?)\s*\n\n"

        sub_question = prompt[-1]['content']
        match = re.search(pattern, sub_question, re.DOTALL)
        extracted_question = match.group(1)

        return extracted_question

    def generate_prompt(self, input_infos, instruction, is_sub_task=False) -> str:

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
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                if is_sub_task: 
                    input_infos_text += f'Related original question:\n\n{content}. \n\nRelated sub-task questions and answers:\n\n'
                else:
                    input_infos_text += f'{content}\n\n'
            elif iteration_idx != -1:
                if is_sub_task and prompt is not None: 
                    extracted_question = self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question
                    else:
                        input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'

                else:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                if is_sub_task and prompt is not None: 
                    extracted_question = self.extract_pattern(prompt)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question # we do not want to duplicate the prompt
                    else:
                        input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        if is_sub_task: 


            if global_format_choice == 'json':

                # prompt = input_infos_text + f'Given the above, answer the following question: {instruction}\n\nIf the question is too complicated or informaion is missing, you still need to give your best answer but add (1) an additional mark [TOO_HARD] in the next line of your final answer (2) information request or decomposison suggestion in the next line of the [TOO_HARD] mark, in the "answer" entry (for example, 300\n[TOO_HARD]\nSuggestion:...) and justify why you think so in the "thinking" entry'# instruction (sub-task in above)
                prompt = input_infos_text + f'Given the above, answer the following question: {instruction} \n\n then justify completely and detailedly why you think so in the "thinking" entry. Again, your task is only to answer the question {instruction} and explaination.'# instruction (sub-task in above)
                prompt += '''\nIf the output require generate `code`, ensure that the final output is stored in variable `result`. 
Eg: 
     
```python           
def calculator(...):
    
    return ...

result = calculator(...)
```
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
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False) -> dict:
        

        system_prompt, prompt = self.generate_prompt(input_infos, instruction, is_sub_task=is_sub_task)

        prompt = [
            _pack_message(content=system_prompt, role="system"),
            _pack_message(content=prompt, role="user")]
        # use system prompt

        response_json = get_json_response_from_gpt(prompt, self.model, self.output_fields, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            print("Key: ", key, " Value: ", value)
            if key == 'code':
                namespace = {}
                print("Generated Code: ", value)
                exec(value, namespace)

                # Truy cập kết quả từ namespace
                output = namespace['result']
                value = output
            info = Info(key, self.__repr__(), value, prompt, None, None, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx,  is_sub_task=is_sub_task)

class AgentSystem():
    """
    Define the Agent System Class. It will call the LLM Agent Class following the MAS (the forward function).
    """
    def __init__(self) -> None:
        pass

    def make_final_answer(self, thinking, answer, sub_tasks=None, agents=None):

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
        elif agents is not None:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, None, '\n'.join(agents), iteration_idx)
        else:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', prompt, '\n'.join(sub_tasks), '\n'.join(agents), iteration_idx)
        return final_answer

    async def forward(self, taskInfo):
        from collections import Counter
        global_max_sc = get_global("global_max_sc")
        global_max_round = get_global("global_max_round")
        global_node_model = get_global("global_node_model")
        global_debate_role = get_global("global_debate_role")
        print("Global node model: ", global_node_model)
        
        # Initialize lists to keep track of sub-tasks and agents
        sub_tasks = []
        agents = []

        # Sub-task 1: Calculate Aya's walking speed, s
        cot_instruction = "Sub-task 1: Calculate Aya's walking speed s based on the information that when she walks at s km/h, the walk takes 4 hours including t minutes spent in the coffee shop."
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
        thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
        agents.append(f'CoT agent {cot_agent.id}, calculating speed s, thinking: {thinking1.content}; answer: {answer1.content}')
        sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

        # Sub-task 2: Determine the time spent in the coffee shop, t
        cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, determine the time spent in the coffee shop t using the information that when she walks at s+2 km/h, the walk takes 2 hours and 24 minutes including t minutes in the coffee shop."
        N = global_max_sc
        cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.5) for _ in range(N)]
        possible_answers = []
        thinking_mapping = {}
        answer_mapping = {}
        for i in range(N):
            thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
            agents.append(f'CoT-SC agent {cot_agents[i].id}, determining time t, thinking: {thinking2.content}; answer: {answer2.content}')
            possible_answers.append(answer2.content)
            thinking_mapping[answer2.content] = thinking2
            answer_mapping[answer2.content] = answer2
        answer2 = Counter(possible_answers).most_common(1)[0][0]
        thinking2 = thinking_mapping[answer2]
        answer2 = answer_mapping[answer2]
        sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

        # Sub-task 3: Calculate the time it takes for Aya to walk 9 km at s+1/2 km/h
        pot_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, write Python code to calculate the time it takes for Aya to walk 9 km at s+1/2 km/h, including the t minutes spent in the coffee shop."
        pot_agent = LLMAgentBase(['thinking', 'code'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
        thinking3, answer3 = cot_agent([taskInfo], pot_instruction, is_sub_task=True)
        agents.append(f'PoT agent {pot_agent.id}, calculating the time it takes for Aya to walk 9 km at s+1/2 km/h, thinking: {thinking3.content}; answer: {answer3.content}')
        sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

        print("Final answer: ", sub_tasks[-1])

        # Generate final consolidated answer
        final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
        return final_answer

if __name__ == "__main__":
    
    global_task_queue = [
        Info(
            "task", 
            "User", 
            """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.""", 
            None, 
            None, 
            None, 
            -1
        )
    ]

    global_node_model = "gpt-4o_chatgpt"
    cot_instruction = "Please think step by step and then solve the task."
    dataset = "workflow_search/aime24"
    technique = dataset.split('/')[0] 
    data_scorer = DataScorer(dataset, technique)
    # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
    output_description = "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."

    debate_role = ['Math Professor', 'Grade School Teacher']
    FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n"""
    model_sampler_map = {
        "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
            model="gpt-4o-mini",
        ),
        "gpt-4o_chatgpt": ChatCompletionSampler(
            model="gpt-4o",
        ),
    }
    set_global("global_score_compute", data_scorer.score)
    
    set_global("global_model_sampler_map", model_sampler_map)
    set_global("global_COST_TOTAL", 0.0)
    set_global("global_format_choice", 'json')
    set_global("global_FORMAT_INST", FORMAT_INST)
    set_global("global_node_model", global_node_model)
    set_global("global_output_description", output_description)
    set_global("global_max_round", 5)
    set_global("global_max_sc", 5)
    set_global("global_debate_role", debate_role)
    set_global("global_task_queue", global_task_queue)
    
    questions = [["Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."]]
    answers = [["204"]]
    set_global("global_answers", answers)
    set_global("global_questions", questions)
    set_global("global_cot_instruction", cot_instruction)
    agentSystem = AgentSystem()
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, global_task_queue), total=len(global_task_queue)))
    
    # dataset = load_dataset("simplescaling/aime24_nofigures")
    # df = pd.DataFrame(dataset['train'])
    # examples = [row.to_dict() for _, row in df.iterrows()]

    # for example_id,example in enumerate(examples):
    #     instance_id = example_id

    #     questions = [example['problem']]
    #     answers = [example['answer']]

    #     task_queue = []
    #     for q in questions:
    #         taskInfo = ('task', 'User', q, None, None, None, -1)
    #         task_queue.append(taskInfo)
            
    #     print(questions)