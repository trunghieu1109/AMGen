import argparse
import copy
import json
import os
import random
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio
# import traceback
# from FlagEmbedding import BGEM3FlagModel

from sampler.chat_completion_sampler import ChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler

import backoff
import numpy as np
import openai
from tqdm import tqdm
import types
import json
import time

import re
import spacy

from typing import Any, List
from datasets import load_dataset
import pandas as pd
import common
from common import HTML_JINJA, get_init_archive, get_prompt, get_reflexion_prompt, SingleEvalResult, get_reflexion_after_eval
from common import get_json_response_from_gpt, get_json_response_from_gpt_reflect, _pack_message, get_embeddings
from utils import random_id, bootstrap_confidence_interval
from common import ANSWER_PATTERN, shorten_context, merge_context
import copy
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from utils import  extract_xml
from shared_vars import set_global, get_global
from pathlib import Path
import asyncio

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

ABSTRACTED_WORKFLOW_TEMPLATE = '''
async def forward(self, taskInfo):
   
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    """
    [Stage 1: <Fill the stage 1's stage_name>]
    
    [Objective] 
    - <Describe in detail the abstracted objective of stage 1.>
    - <Describe in detail the abstracted objective of stage 1.>
    - <Describe in detail the abstracted objective of stage 1.>
    
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage 1.>
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    # Sub-task 1: Analyze first expression/data component with self-consistency
    cot_instruction = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, analyzing [expression #1], thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    
    print("Subtask 1 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 2: <Fill the stage 2's stage_name>]
    
    [Objective] 
    - <Describe in detail the abstracted objective of stage 2.>
    - <Describe in detail the abstracted objective of stage 2.>
    - <Describe in detail the abstracted objective of stage 2.>
    
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage 2.>
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    # Sub-task 2: Calculate intermediate output with reflexion
    cot_reflect_instruction = "Sub-task 2: Based on Sub-task 1 outputs, calculate intermediate values and synthesize key insights"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input aggregation from previous stages
    cot_inputs = [taskInfo, thinking1, answer1]
    
    # Generate initial intermediate computation
    thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {{cot_agent.id}}, calculating intermediate output, thinking: {{thinking2.content}}; answer: {{answer2.content}}")

    # Iterative refinement through critic feedback
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], 
                                       "Critically evaluate the [intermediate calculation], mathematical correctness, and completeness and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {{critic_agent.id}}, providing feedback, thinking: {{feedback.content}}; answer: {{correct.content}}")
        if correct.content == "True":
            break
        
        # Incorporate feedback for next iteration
        cot_inputs.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {{cot_agent.id}}, refining intermediate output, thinking: {{thinking2.content}}; answer: {{answer2.content}}")
    
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}}; answer - {{answer2.content}}")
    
    print("Subtask 2 answer: ", sub_tasks[-1])
    
    # --------------------------------------------------------------------------------------------------------------
    
    """
    [Stage 3: <Fill the stage 3's stage_name>]
    
    [Objective] 
    - <Describe in detail the abstracted objective of stage 3.>
    - <Describe in detail the abstracted objective of stage 3.>
    - <Describe in detail the abstracted objective of stage 3.>
    
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage 3.>
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    # Sub-task 3: Calculate the [final output]
    debate_instruction_3 = "Sub-task 3: Based on the output of sub-tasks 1 and 2, calculate the [final output], with context ...."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    
    for r in range(N_max_3):
        # Multiple rounds of debate allow agents to build on each other"s reasoning.
        for i, agent in enumerate(debate_agents_3):
            input_infos_3 = [taskInfo, thinking2, answer2]
            if r > 0:
                input_infos_4.extend(all_thinking3[r-1])
            thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating [final output], thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
            
    # A final decision agent synthesizes the debate into a single, conclusive answer.
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make a final decision on the [final output].", is_sub_task=True)
    agents.append(f"Final Decision agent on calculating [final output], thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    
    print("Subtask 3 answer: ", sub_tasks[-1])
    
    <Continue with next stages>
    
    """
    [Stage n: <Fill the stage n's stage_name>]
    
    [Objective] 
    - <Describe in detail the abstracted objective of stage n.>
    - <Describe in detail the abstracted objective of stage n.>
    - <Describe in detail the abstracted objective of stage n.>
    
    [Agent Collaborations]
    - <Describe in detail the agent collaboration of stage n.>
    
    [Examples]
    - Here are many examples of implementing subtasks in this stage.
    - It could be more than 2 subtasks in this stage, based on your fine-grained task decomposition.
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    final_answer = await self.make_final_answer(thinkingn, answern, sub_tasks, agents)
    return final_answer
'''

AGENT_INTERACTION_PATTERN = '''
Chain-of-Thought: 
```python
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, consider/calculate all possible scenarios of [problem #1], thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
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
    
    for i in range(N):
        # Each CoT-SC agent tries to calculate all possible cases independently
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}}, consider all possible cases of [problem #2], thinking: {{thinking2.content}}; answer: {{answer2.content}}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
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
```
'''

def levenshtein_array(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m+1) for _ in range(n+1)]

    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0:
                dp[i][j] = max(i, j)
            else:
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,     # delete
                    dp[i][j-1] + 1,     # insert
                    dp[i-1][j-1] + cost # substitute
                )
    return dp[n][m]

def levenshtein_array_to_array(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i  # deletion
    for j in range(m + 1):
        dp[0][j] = j  # insertion

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m]


class Subtask:
    subtask_id: str
    objective: str
    dependencies: list[str]
    agent_collaboration: str # CoT, SC_CoT, Debate, Reflexion
    
class AbstractedSubtask:
    subtask_id: str
    abstracted_objective: str
    objective: list[str] = []
    name: str = ""
    dependencies: list[str] = []
    agent_collaboration: list[str] = []
    
    def to_dict(self):
        return {
            "subtask_id": self.subtask_id,
            "abstracted_objective": self.abstracted_objective,
            "objective": self.objective,
            "name": self.name,
            "dependencies": self.dependencies,
            "agent_collaboration": self.agent_collaboration
        }

class Utils:
    
    def __init__(self) -> None:
        pass
    
    def read_json_from_directory(self, dir_path):
        
        json_file = {}

        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                filepath = os.path.join(dir_path, filename)
                if 'iteration_0' in filepath:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        json_file[filename] = data
                    
        return json_file
    
    def preprocess(self, subtask: str):
        
        # lower subtask description
        subtask = subtask.lower()
        # remove punctuation and special characters 
        subtask = re.sub(r'[^\w\s]', '', subtask)
        subtask = re.sub(r'\s+', ' ', subtask).strip()
        lemmatizer = WordNetLemmatizer()
        subtask = ' '.join([lemmatizer.lemmatize(w) for w in subtask.split()])
        
        return subtask
    
    def are_similar(self, s1, s2, threshold=1):
        return distance(s1, s2) <= threshold

    def merge_workflow(self, workflow_set):
        clusters = []
        for seq in workflow_set:
            merged = False
            for cluster in clusters:
                # Kiểm tra nếu seq tương tự với *tất cả* chuỗi trong cụm
                if all(self.are_similar(seq, other) for other in cluster):
                    cluster.append(seq)
                    merged = True
                    break
            if not merged:
                clusters.append([seq])
        return clusters
    
    def merge_sequences_aligned(self, sequences):
        # Chuyển chuỗi thành list ký tự
        char_lists = [list(seq) for seq in sequences]
        
        # Tìm chiều dài ngắn nhất để bắt đầu so từ cuối
        min_len = min(len(seq) for seq in char_lists)
        merged = []

        # So sánh từ cuối về đầu
        for i in range(1, min_len + 1):
            chars_at_pos = [seq[-i] for seq in char_lists if len(seq) >= i]
            unique_chars = list(dict.fromkeys(chars_at_pos))  # giữ thứ tự gốc
            if len(unique_chars) == 1:
                merged.insert(0, unique_chars[0])
            else:
                merged.insert(0, unique_chars)

        # Xử lý phần đầu còn lại (phần thêm ở đầu của các chuỗi dài hơn)
        max_len = max(len(seq) for seq in char_lists)
        for seq in char_lists:
            if len(seq) > min_len:
                for i in range(len(seq) - min_len):
                    merged.insert(0, [seq[i]])

        return merged
    
    def get_most_frequent(self, object_list, top_k):
        from collections import Counter

        counter = Counter(object_list)
        most_common_two = counter.most_common(top_k)
        return [item for item, _ in most_common_two]

    def levenshtein_array(self, a, b):
        n, m = len(a), len(b)
        dp = [[0] * (m+1) for _ in range(n+1)]

        for i in range(n+1):
            for j in range(m+1):
                if i == 0 or j == 0:
                    dp[i][j] = max(i, j)
                else:
                    cost = 0 if a[i-1] == b[j-1] else 1
                    dp[i][j] = min(
                        dp[i-1][j] + 1,     # delete
                        dp[i][j-1] + 1,     # insert
                        dp[i-1][j-1] + cost # substitute
                    )
        return dp[n][m]

class MASAbstraction():

    def __init__(self) -> None:
        self.abstracted_cluster_subtask = {}
        self.abstracted_objectives = []
        self.subtask_names = []
        self.subtask_names_set = set()
        self.utils = Utils()
        self.abstracted_tasks = {}
        
    async def analyze(self, query: str, mas: str):

        print("================== Extracting Workflow ==================")

        analysis_prompt = f"""
You are an expert LLM assistant specialized in analyzing multi-agent system (MAS) workflows. Given a natural language query and a concrete abstract workflow, your task is to return a structured JSON analysis that includes:

- Objective: Clearly state the full objective of each subtask in the workflow. Do not include phrases like "Based on subtask..." — state the purpose directly and completely.
- Supporting Information: List any assumptions, context, or input required by each subtask to achieve its objective.
- Agent Collaboration: Specify the reasoning pattern used in the subtask: one of CoT (Chain-of-Thought), SC_CoT (Self-Consistency CoT), Debate, or `Reflexion).
- Dependency: Identify which previous subtasks the current subtask depends on. Use subtask IDs (e.g., "subtask_1") in a list.

[Query]
{query}

[Abstract Workflow]
{mas}

Return your result in valid JSON format with the following structure:
{{
    "thought": "....."
    "subtasks": [
        {{
            "subtask_id": "subtask_1",
            "objective": "",
            "supporting_info": "",
            "agent_collaboration": CoT | SC_CoT | Debate | Reflexion,
            "dependencies": []
        }},
        {{
            "subtask_id": "subtask_2",
            "objective": "",
            "supporting_info": "",
            "agent_collaboration": CoT | SC_CoT | Debate | Reflexion,
            "dependencies": ["subtask_1"]
        }},
        .....
        {{
            "subtask_id": "subtask_n",
            "objective": "",
            "supporting_info": "",
            "agent_collaboration": CoT | SC_CoT | Debate | Reflexion,
            "dependencies": ["subtask_x", "subtask_y", ...]
        }}
    ]
}}
    """
    
        msg_list = [
            {"role": "user", "content": analysis_prompt},
        ]

        analysis,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), "gpt-4o-mini-2024-07-18", ['thought', 'subtasks'], 0.0)
        
        print("============== Extracted workflow ==============")
        print(analysis['subtasks'])
        
        return analysis['subtasks']
    
    async def abstract_task_decomposition(self, query, subtasks):
        # generate the first generation
        print("================== Abstracting Workflow ==================")
        
        recent_subtask_names = self.subtask_names_set
        recent_abstracted_objectives = self.abstracted_objectives[-20:] if len(self.abstracted_objectives) >= 20 else self.abstracted_objectives
        
        abstract_td_user_prompt = f"""
You are a reasoning abstraction expert tasked with converting a list of domain-specific reasoning steps (subtasks), originally designed for a specific query, into purely functional, domain-independent, and query-agnostic versions. The abstracted subtasks must be so generalized that they bear no resemblance to the original query or domain, functioning as universal steps applicable to any query in any field. Follow these guidelines for each subtask:

1. Create a Generalized Subtask Name: Assign a concise, domain-independent subtask_name as the title for each subtask.
2. Avoid Domain-Specific Terms: Use logic- or function-based language (e.g., "aggregate input values", "identify constrained group", "derive target variable") instead of domain-specific terms (e.g., union, velocity, kinetic energy).
3. Eliminate Intermediate Constructs: Remove references to domain-specific constructs like rate, distance, duration, set intersection, maximum, or minimum.
4. Remove Query-Specific Context: Eliminate query-specific terms, formulas, adjectives, or contexts (e.g., Pythagoras’ theorem, GDP, 2020).
5. Remove Dependencies on Previous Subtasks: Avoid references to outputs or results from other subtasks (e.g., "Using the output of subtask_3", "Divide value from subtask_2").

Subtask Name Options: Select a `subtask_name` from the following list if it fits to the subtask objective, or create a new one if none are sufficiently abstract:
<subtask_name>
{recent_subtask_names}
</subtask_name>

Reference for Abstraction Style: Review the following abstracted objectives to understand the level of generalization required, but do not copy them unless they meet the purely functional, query-agnostic standard:
<abstracted_objective>
{recent_abstracted_objectives}
</abstracted_objective>

Task: For each provided subtask, generate an abstracted version with a subtask_name and a description that strictly adheres to the above guidelines. The output must be concise, purely functional, and universally applicable, with no trace of the original query or domain.

Return sufficient subtasks, covering all {len(subtasks)} in the original list.

Here is the user query. Ensure the abstracted subtask list must not be related to this query any more:
{query}

Here is subtask list:
{subtasks}

The output is in JSON format and must include:
- thought: A description of your abstraction strategy and how you followed the above steps.
- abstracted_subtasks: A full list of subtasks with their generalized names and objectives. Each element contains:
    + subtask_id
    + abstracted_objective
    + subtask_name

Return the result in the following JSON format in English:
{{
  "thought": "Your overall thought process on how the subtasks were abstracted, following the instructed steps",
  "abstracted_subtasks": [
    {{
      "subtask_id": "subtask_1",
      "abstracted_objective": "Generalized, domain-independent description of the subtask"
      "subtask_name": "A concise and abstract name for this abstracted objective"
    }},
    ...
  ]
}}
        """
        
        msg_list = [
            {"role": "user", "content": abstract_td_user_prompt}
        ]

        abstracted_subtasks,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), "o4-mini", ['thought', 'abstracted_subtasks'], 0.0)
        
        for subtask in abstracted_subtasks['abstracted_subtasks']:
            print(subtask)
            self.abstracted_objectives.append(subtask['abstracted_objective'])
            self.subtask_names.append(subtask['subtask_name'])
            self.subtask_names_set.add(subtask['subtask_name'])
            
        return abstracted_subtasks['abstracted_subtasks']
    
    async def tagged_subtask(self, subtask: str):
        tagged_prompt = f"""
You are a reasoning abstraction expert.

Given the following reasoning subtask, return a short abstracted name (CamelCase) that captures its logical function.

Subtask: "{subtask}"
AbstractedName:

Return your result in valid JSON format with the following structure:
{{
  "thought": "Explain your reasoning process, including how the subtasks were interpreted and the shared logic you abstracted.",
  "subtask_name": "A short abstracted name that captures subtask's logical function"
}}
        """
        
        msg_list = [
            {"role": "user", "content": tagged_prompt},
        ]

        subtask_name,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), "gpt-4.1-mini", ['thought', 'subtask_name'], 0.0)

        print(f"Subtask: {subtask}, Subtask name: {subtask_name['subtask_name']}")
        
        return subtask_name['subtask_name']
    
    async def embedding_each_subtask(self, subtask):

        embedding = self.model.encode(subtask, normalize_embeddings=True)
        
        return embedding
    
    async def embedding_subtask(self, subtasks: List[str]):
        print("Input length: ", len(subtasks))
        preprocessed_subtasks = [self.utils.preprocess(subtask) for subtask in subtasks]

        # embeddings = [await self.embedding_each_subtask(subtask) for subtask in subtasks]
        # embeddings = self.bge_model.encode(subtasks)['dense_vecs']
        embeddings = await get_embeddings(subtasks, "text-embedding-3-large")
        
        print("Output length: ", len(embeddings))
        
        return embeddings
    
    async def visualize_embeddings(self, embeddings, method='pca'):
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

        reduced = reducer.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=10)
        plt.title(f'Embedding Visualization using {method.upper()}')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.savefig("clustering/embedding_visualization.png")
        # plt.show()
        
    async def find_optimal_k(self, embeddings):
        pca = PCA()
        pca.fit(embeddings)

        # explained_variance_ratio_: tỉ lệ phương sai mỗi thành phần chính
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_variance, marker='o')
        plt.xlabel('Số chiều sau khi giảm')
        plt.ylabel('Tổng phương sai tích lũy')
        plt.title('Chọn số chiều tối ưu với PCA')
        plt.grid(True)
        plt.axhline(y=0.95, color='r', linestyle='--')  # Ngưỡng 95%
        plt.axhline(y=0.9, color='b', linestyle='--')  # Ngưỡng 95%
        plt.savefig("clustering/optimal_dimension_nums.png")
        
        dim_90 = np.argmax(cumulative_variance >= 0.90) + 1
        dim_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print("Dim 90 and 95%: ", dim_90, dim_95)

        return dim_90, dim_95
        
    async def clustering(self, embeddings):
        sse = []  # Sum of Squared Errors
        k_range = range(1, 100)
        
        normalized_embeddings = normalize(embeddings, norm="l2")
        dim_90, dim_95 = await self.find_optimal_k(normalized_embeddings)
        
        # reduce dimentions from 384 -> 50 to clustering
        pca_50 = PCA(n_components=dim_95, random_state=42)
        reduced_embeddings = pca_50.fit_transform(normalized_embeddings)
        
        print(sum(pca_50.explained_variance_ratio_))
        
        # find optimal k_nums
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(reduced_embeddings)
            sse.append(kmeans.inertia_)
            
        knee = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
        optimal_k = knee.knee
        
        plt.plot(k_range, sse, 'bo-')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Sum of Squared Errors (SSE)")
        plt.title("Elbow Method for Optimal k")
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.legend()
        plt.grid(True)
        plt.savefig("clustering/elbow_method_for_optimal_k.png")
        print("Optimal k: ", optimal_k)
        # clustering 
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(reduced_embeddings)

        use_tsne = False

        if use_tsne:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            reduced_2d = tsne.fit_transform(reduced_embeddings)
        else:
            pca_2 = PCA(n_components=2)
            reduced_2d = pca_2.fit_transform(reduced_embeddings)

        # === Bước 6: Vẽ kết quả clustering ===
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c=labels, cmap='tab10', s=15)
        plt.title(f"KMeans Clustering Visualization (k = {optimal_k})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.savefig("clustering/optimal_k_clustering.png")
        
        return kmeans, pca_50, labels, reduced_embeddings
    
    async def agglomerative_clustering(self, embeddings):
        
        # normalize embedding vectors
        await self.find_optimal_k(embeddings)
        
        scaled = StandardScaler().fit_transform(embeddings)
        pca_50 = PCA(n_components=80, random_state=42)
        reduced_embeddings = pca_50.fit_transform(scaled)
        
        distances = []

        ks = range(2, 15)
        for k in ks:
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(reduced_embeddings)
            
            intra_dists = []
            for label in set(labels):
                cluster_points = reduced_embeddings[labels == label]
                if len(cluster_points) > 1:
                    dist = pairwise_distances(cluster_points)
                    mean_dist = np.mean(dist)
                    intra_dists.append(mean_dist)
            
            avg_intra_cluster_dist = np.mean(intra_dists)
            distances.append(avg_intra_cluster_dist)      
        
        
        knee = KneeLocator(ks, distances, curve='convex', direction='decreasing')
        print("Elbow at k =", knee.elbow)  

        clustering = AgglomerativeClustering(
            n_clusters=knee.elbow,
        )
        
        labels = clustering.fit_predict(reduced_embeddings)
        
        return labels, reduced_embeddings
    
    async def aggregate_cluster_subtask(self, subtasks: List[str]):

        # print("================== Aggregate Cluster Subtask ==================")

        aggregate_prompt = f"""
You are an expert LLM assistant specializing in analyzing and abstracting multi-agent system (MAS) workflows. Your task is to review a given list of natural language subtasks and synthesize a single generalized subtask that encapsulates their core logical operation in a domain-agnostic way.

# The generalized subtask must:
- Be reusable across diverse domains (e.g., science, mathematics, software engineering, logistics, healthcare, etc.)
- Abstract the underlying reasoning function shared by all the provided subtasks
- Eliminate any domain-specific terms or context
- Fully reflect the main objective of each subtask in the list
- Focus on the nature meaning of each subtask, ensure generalized subtasks must cover the nature meaning of each subtask.
- Beside the generalized_subtask, return a `subtask_name`, which is concise and general for every subtask in subtask's list.

# Input:
Subtask's list:
{subtasks}

# Output:
Return your result in valid JSON format with the following structure:
{{
  "thought": "Explain your reasoning process, including how the subtasks were interpreted and the shared logic you abstracted.",
  "generalized_subtask": "A domain-neutral statement representing the common purpose of all subtasks."
  "subtask_name": "A concise and general subtask_name for every subtask in subtask's list"
}}
Be precise, logical, and abstraction-focused.
        """
    
        msg_list = [
            {"role": "user", "content": aggregate_prompt},
        ]

        generalized_subtask,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), "gpt-4.1-mini", ['thought', 'generalized_subtask', 'subtask_name'], 0.0)

        print(f"Generalized subtask: {generalized_subtask['generalized_subtask']}")

        return generalized_subtask['generalized_subtask'], generalized_subtask['subtask_name'],
    
    async def clustering_subtasks_list(self, workflow_path: str):
        
        c2s_exist = os.path.exists(f"{workflow_path}/cluster_to_subtask_mapping.json")
        s2c_exist = os.path.exists(f"{workflow_path}/subtask_to_cluster.json")
        kmeans_exist = os.path.exists(f"{workflow_path}/kmeans.pkl")
        pca_exist = os.path.exists(f"{workflow_path}/pca.pkl")
        ab_exist = os.path.exists(f"{workflow_path}/abstracted_subtasks.pkl")
        mas_chain_exist = os.path.exists(f"{workflow_path}/mas_chain.json")
        
        if c2s_exist and s2c_exist and kmeans_exist and mas_chain_exist and ab_exist and pca_exist:
            with open(f"{workflow_path}/cluster_to_subtask_mapping.json", "r", encoding="utf-8") as f:
                cluster_to_subtask = json.load(f)

            with open(f"{workflow_path}/subtask_to_cluster.json", "r", encoding="utf-8") as f:
                subtask_to_cluster = json.loads(f.read())

            with open(f"{workflow_path}/kmeans.pkl", "rb") as f:
                kmeans = pkl.load(f)
                
            with open(f"{workflow_path}/pca.pkl", "rb") as f:
                pca = pkl.load(f)
                
            with open(f"{workflow_path}/abstracted_subtasks.pkl", "rb") as f:
                abstracted_subtasks_list = pkl.load(f)

            with open(f"{workflow_path}/mas_chain.json", "r", encoding="utf-8") as f:
                mas_chain = json.loads(f.read()) 
        
            return cluster_to_subtask, subtask_to_cluster, kmeans, pca, abstracted_subtasks_list, mas_chain 
        
        mas_zero_json = self.utils.read_json_from_directory(workflow_path)
    
        subtasks = {
            'objective': [],
            'subtask_name': [],
            'subtask_abstracted_objective': [],
            'merge_subtask': [],
            'clusters': [],
            'agent_collaboration': [],
            'dependencies': []
        }
        
        clusters_to_subtasks = {}
        
        for k, v in mas_zero_json.items():
            try:
                for subtask in v:
                    # print(k, " ", "subtask: ", subtask['objective'])
                    subtasks['objective'].append(subtask['objective'])
                    subtasks['subtask_name'].append(subtask['subtask_name'])
                    subtasks['subtask_abstracted_objective'].append(subtask['abstracted_objective'])
                    subtasks['merge_subtask'].append(f"{subtask['subtask_name']}: {subtask['abstracted_objective']}")
                    subtasks['agent_collaboration'].append(subtask['agent_collaboration'])
            except Exception as e:
                # print("Key: ", k)
                continue
            
        # embedding tasks
        embeddings = await self.embedding_subtask(subtasks['merge_subtask'])
        
        # visualize embeddings
        await self.visualize_embeddings(embeddings)
        
        # clustering embeddings vectors
        kmeans, pca, labels, reduced_embeddings = await self.clustering(embeddings)
        # labels, reduced_embeddings = await self.agglomerative_clustering(embeddings)
        
        # assign cluster_id to each subtask
        max_cluster_id = 0
        for idx, subtask in enumerate(subtasks['merge_subtask']):
            cluster = labels[idx]
            max_cluster_id = max(max_cluster_id, cluster)
            subtasks['clusters'].append(str(cluster))
            
        start_idx = 0
        
        mas_set = set()
        
        # merge cluster_ids of each subtask to its workflow -> mas_chain
        for filename, mas in mas_zero_json.items():
            
            mas_chain = []
            inner_cluster = {}
            try:
                for subtask in mas:
                    inner_dependencies = []
                    abstracted_objective = subtask['abstracted_objective']
                    inner_cluster[subtask['subtask_id']] = subtasks['clusters'][start_idx]
                    
                    if abstracted_objective != mas[-1]['abstracted_objective']:
                        if int(subtasks['clusters'][start_idx]) not in mas_chain:
                            mas_chain.append(int(subtasks['clusters'][start_idx]))
                    else:
                        if not mas_chain or int(subtasks['clusters'][start_idx]) != mas_chain[-1]:
                            mas_chain.append(int(subtasks['clusters'][start_idx]))
                        
                    for dep in subtask['dependencies']:
                        subtask_cluster_id = f"subtask_{inner_cluster[dep]}"
                        inner_dependencies.append(subtask_cluster_id)
                        
                    subtasks['dependencies'].append(inner_dependencies)
                    # print(inner_dependencies)
                        
                    start_idx += 1
                if tuple(mas_chain) not in mas_set:
                    print("Filename: ", filename, " ", mas_chain)
                    
                mas_set.add(tuple(mas_chain))
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        print("Clustered set: ", mas_set)
        print("Set length: ", len(mas_set))
        print("Len Clusters: ", len(subtasks['clusters']))
        print("Len Dependencies: ", len(subtasks['dependencies']))

        for idx, cluster_id in enumerate(subtasks['clusters']):
            if str(cluster_id) not in self.abstracted_tasks:
                self.abstracted_tasks[str(cluster_id)] = []
                
            self.abstracted_tasks[str(cluster_id)].append({
                'subtask_name': subtasks['subtask_name'][idx],
                'objective': subtasks['merge_subtask'][idx],
                'agent_collaboration': subtasks['agent_collaboration'][idx],
                'dependencies': subtasks['dependencies'][idx]
            })         
            
        abstracted_subtasks_list = []    
            
        for i in range(max_cluster_id + 1):
            print("================== Aggregate Cluster Subtask ==================")
            print("Cluster: ", i, ' ', len(self.abstracted_tasks[str(i)]))
            ab_subtask = AbstractedSubtask()
            ab_subtask.subtask_id = f"subtask_{i}" 
            
            ab_subtask.objective = []
            ab_subtask.agent_collaboration = []
            ab_subtask.dependencies = []
            dependencies_set = set()
            for subtask in self.abstracted_tasks[str(i)]:
                print(subtask['objective'])
                ab_subtask.objective.append(subtask['objective'])
                ab_subtask.agent_collaboration.append(subtask['agent_collaboration'])
                for dep in subtask['dependencies']:
                    dependencies_set.add(dep)
            ab_subtask.dependencies.append(list(dependencies_set))    
            ab_subtask.agent_collaboration = self.utils.get_most_frequent(ab_subtask.agent_collaboration, 2)
            cluster_subtask, cluster_subtask_name = await self.aggregate_cluster_subtask(self.abstracted_tasks[str(i)])
            self.abstracted_cluster_subtask[str(i)] = cluster_subtask
            ab_subtask.abstracted_objective = cluster_subtask
            abstracted_subtasks_list.append(ab_subtask)
            ab_subtask.name = cluster_subtask_name
            
        mas_set_list = []
        for mas in mas_set:
            mas_ = []
            for id in list(mas):
                mas_.append(str(id))
            mas_set_list.append(mas_)
            
        with open(f"{workflow_path}/cluster_to_subtask_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.abstracted_cluster_subtask, f, ensure_ascii=False, indent=4) 
            
        with open(f"{workflow_path}/subtask_to_cluster.json", "w", encoding="utf-8") as f:
            json.dump(subtasks, f, ensure_ascii=False, indent=4)
            
        with open(f"{workflow_path}/kmeans.pkl", "wb") as f:
            pkl.dump(kmeans, f) 
            
        with open(f"{workflow_path}/pca.pkl", "wb") as f:
            pkl.dump(pca, f) 
            
        with open(f"{workflow_path}/abstracted_subtasks.pkl", "wb") as f:
            pkl.dump(abstracted_subtasks_list, f) 
            
        with open(f"{workflow_path}/mas_chain.json", "w", encoding="utf-8") as f:
            json.dump(mas_set_list, f, ensure_ascii=False, indent=4)
            
        # return None, None, None, None, None, None
        return self.abstracted_cluster_subtask, subtasks, kmeans, pca, abstracted_subtasks_list, mas_set_list 
            
    async def merge_each_group(self, cluster_to_subtask, cluster_to_agent_collaboration, cluster_to_subtask_name, mas_chain):
        # print("================== Aggregate Cluster Subtask ==================")
        # print(mas_chain)
        
        if len(mas_chain) == 1:
            # handle for single mas_chain
            merged_chain = []
            merged_workflow = []
            for idx, subtask_id in enumerate(mas_chain[0]):
                merged_chain.append(str(subtask_id))
                merged_workflow.append({
                    "subtask_id": f"subtask_{idx}",
                    "subtask_name": cluster_to_subtask_name[str(subtask_id)],
                    "agent_collaboration": cluster_to_agent_collaboration[str(subtask_id)],
                    "abstracted_objective": cluster_to_subtask[str(subtask_id)],
                })
                
                # print(f"subtask_{idx}: {subtask_id}")s
            
            return merged_workflow, merged_chain

        workflows = [f"Workflow {idx + 1}: {mas}" for idx, mas in enumerate(mas_chain)]
        workflows_desc = "\n".join(workflows)
        c2s = [f"Subtask {k}: {v}" for k, v in cluster_to_subtask.items()]
        c2s_desc = "\n".join(c2s)
        c2ac = [f"Agent Collaboration for Subtask {k}: {v}" for k, v in cluster_to_agent_collaboration.items()]
        c2ac_desc = "\n".join(c2ac)
        
        c2sn = [f"Subtask name for Subtask {k}: {v}" for k, v in cluster_to_subtask_name.items()]
        c2sn_desc = "\n".join(c2sn)

        merging_prompt = f"""
You are tasked with merging workflow chains into a single, cohesive workflow while adhering to specific requirements. 
The merged workflow must be a sequence of subtasks, presented as individual steps, with optional subtasks clearly marked, ensuring all dependencies and constraints are satisfied.

# Subtask Definitions
Each subtask ID corresponds to a specific operation:
{c2s_desc}

# Subtask name
Each subtask has a concises name as follow:
{c2sn_desc}

# Agent Collaboration
Each subtask corresponds to a list of the most frequent agent collaboration patterns:
{c2ac_desc}

# Input Workflows
The workflows, each represented as a sequence of subtask IDs, are:
{workflows_desc}

Then, your task is to merge above workflow to create a single, cohesive workflow that covers all the provided workflow.

# Here are instructions to merge workflows, steps-by-steps:
1. Consider the objective of each subtask in the workflows thoroughly and perceive their relationships.
2. The merged workflow has maximum 6 subtasks.
3. Preserve Execution Order: Maintain the sequential order of subtasks within each workflow. Respect the relative order of workflows as presented when positioning unique subtasks, unless dependencies dictate otherwise.
4. Handle Common Subtasks: Subtasks appearing in multiple workflows should be executed exactly once in the merged workflow, placed to satisfy all workflows that include them.
5. Merge Less Frequent Subtasks: For the less frequent subtasks, incorporate them into the workflow at their correct relative positions with respect to the common subtasks. If multiple subtasks appear at the same position, merge them into a concise and synthesized subtask. In this case, `subtask_name` also merged to only a combined `subtask_name`
For example, if workflow A is ['1', '2', '6'] and workflow B is ['1', '2', '3'], the merged chain is ['1', '2', ['6', '3']]. And in the merged workflow, the subtask 3 is combined to subtask 6 to create a concise and synthesized subtask.
6. Expected Output Format
Provide the thought and the merged workflow in the JSON format:
- thought: Explain your reasoning process, including how provided workflow are merged into the single synthesized one
- merged_chain: The merged chain, which is combined from provided workflow chains.
- merged_workflow: A numbered list of steps, with each step including:
    + `subtask_id`: It is not the original id of this subtask. It is the new id of this subtask in the merged workflow, start from `subtask_0`.
    + `subtask_name`: The corresponding subtask name for this subtask.
    + `merged_chain`: The merged chain, which is combined from provided workflow chains.
    + `agent_collaboration`: The agent collaboration patterns that are frequently used.
    + `abstracted_objective`: The objective of this subtask, after merging from multiple workflows. 

# Important nodes:
When merging multiple subtasks, they must be grouped into a list. So result is List[string] or List[str or List[str]]. The string is only the digit value, not a string of a list.
Example: 
- Wrong Cases: ['[4,6]', '11', '7']
- Correct Cases: [['4','6'], '11', '7']
# Output:
Return your result in valid JSON format with the following structure:
{{
  "thought": "Explain your reasoning process, including how provided workflow are merged into the single synthesized one",
  "merged_chain": ['idx1', 'idx2', 'idx3', 'idx4', ....]
  "merged_workflow": [
      {{
          "subtask_id": "subtask_1",
          "subtask_name": "The corresponding subtask name for this subtask.",
          "agent_collaboration": [CoT | SC_CoT | Reflexion | Debate],
          "abstracted_objective": ""
      }},
      {{
          "subtask_id": "subtask_2",
          "subtask_name": "The corresponding subtask name for this subtask.",
          "agent_collaboration": [CoT | SC_CoT | Reflexion | Debate],
          "abstracted_objective": ""
      }},
      ...
      {{
          "subtask_id": "subtask_n",
          "subtask_name": "The corresponding subtask name for this subtask.",
          "agent_collaboration": [CoT | SC_CoT | Reflexion | Debate],
          "abstracted_objective": ""
      }},
  ]
}}
        """
    
        msg_list = [
            {"role": "user", "content": merging_prompt},
        ]

        merged_workflow ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), "o4-mini", ['thought', 'merged_chain', 'merged_workflow'], 0.0)

        return merged_workflow['merged_workflow'], merged_workflow['merged_chain'] 
    
    async def generate_abstracted_workflow(self, merged_workflow):

        merging_prompt = f"""
You are a smart assistant. Your task is to create an agentic workflow implemented as a Python forward function that processes a task through multiple stages, leveraging agent collaboration patterns such as Chain-of-Thought (CoT), Self-Consistency Chain-of-Thought (SC_CoT), Reflexion, and Debate. The workflow should handle a given taskInfo input, decompose the task into stages and subtasks, and produce a final consolidated answer. Use the provided stage descriptions and agent collaboration patterns to guide the implementation.

Here are the stages's description: 
{merged_workflow}

To address subtasks in each stage, you must apply the following agent collaboration patterns: Chain-of-Thought, Self-Consistency Chain-of-Thought, Debate and Reflexion. Below are the implementation of them.
{AGENT_INTERACTION_PATTERN} 

# Instruction:
- Function Signature: Implement an async def forward(self, taskInfo) function.
- Structure:
    + Initialize lists sub_tasks and agents to track subtask outputs and agent interactions.
    + For each stage, include:
        - A detailed description of the stage's objectives, agent collaborations, and example subtasks.
        - Implement only 1 sample subtask with only one suggested agent collaboration patterns.
    + Print subtask outputs for debugging.
    + Generate a final answer using self.make_final_answer.
- Subtask Implementation:
    + Decompose each stage into fine-grained subtasks (at least one per stage, not required by two).
    + Use the provided agent collaboration patterns for each stage.
    + Ensure subtasks build on previous outputs, forming a logical progression.
    + For SC_CoT, use self.max_sc for the number of agents and select the most common answer via Counter.
    + For Reflexion and Debate, use self.max_round for the number of iterations.
- Output:
    + Return a final consolidated answer via self.make_final_answer(thinking, answer, sub_tasks, agents).
    + Then, your task is to merge above workflow to create a single, cohesive workflow that covers all the provided workflow.
    
# Abstracted Workflow Template:
{ABSTRACTED_WORKFLOW_TEMPLATE}

Return your result in valid JSON format with the following structure:
{{
  "thought": "Explain your reasoning process, including how provided workflow are merged into the single synthesized one",
  "abstracted_workflow": "The generate workflow based on provided information"
}}
        """
    
        msg_list = [
            {"role": "user", "content": merging_prompt},
        ]

        abstracted_workflow ,_ = await get_json_response_from_gpt(copy.deepcopy(msg_list), "gpt-4o_chatgpt", ['thought', 'abstracted_workflow'], 0.0)

        print(abstracted_workflow['abstracted_workflow'])

        return abstracted_workflow['abstracted_workflow']
            
    async def clustering_workflow(self, workflow_path, cluster_to_subtask, cluster_to_agent_collaboration, cluster_to_subtask_name, cluster_to_dependencies, mas_chain):
        abstract_workflow_description = []
        results = {}
        visited = set()
        distince_workflow = 0

        for i, seq1 in enumerate(mas_chain):
            if tuple(seq1) in visited:
                continue

            group = [seq1]
            visited.add(tuple(seq1))

            for j, seq2 in enumerate(mas_chain):
                if i == j or tuple(seq2) in visited:
                    continue

                # Check if seq2 has distance <= 1 with ALL items in current group
                is_similar_to_all = all(self.utils.levenshtein_array(seq2, other) <= 1 for other in group)

                if is_similar_to_all:
                    group.append(seq2)
                    visited.add(tuple(seq2))

            if group:
                distince_workflow += 1
                results[tuple(group[0])] = group[1:]

        print("Distinct workflows:", distince_workflow)
        idx = 0
        merged_workflow_chains = []
        mas_idx = 0
        for key, val in results.items():
            # idx += 1
            # if idx < 13:
            #     continue
            sub_mas_chain = []
            sub_mas_chain.append(list(key))
            for match in val:
                sub_mas_chain.append(match)
                
            print(sub_mas_chain)
            merged_workflow, merged_chain = await self.merge_each_group(cluster_to_subtask, cluster_to_agent_collaboration, cluster_to_subtask_name, sub_mas_chain)
            merged_workflow_chains.append(merged_chain)
            print("Merged chain: ", merged_chain)
            print("Merged Workflow: ", merged_workflow)
            
            os.makedirs(f"{workflow_path}/abstracted_workflow", exist_ok=True)
            idx += 1
            
            abstract_subtask_list = []
            mas_dependencies = {}
            print(merged_chain)
            
            for idx1, item in enumerate(merged_chain):
                mas_dependencies[f'subtask_{idx1}'] = set()
                if idx1 + 1 < len(merged_chain):
                    mas_dependencies[f'subtask_{idx1}'].add(f'subtask_{idx1 + 1}')
                if isinstance(item, list):
                    for idx2 in range(idx1 + 1, len(merged_chain)):
                        if isinstance(merged_chain[idx2], list):
                            mas_dependencies[f'subtask_{idx1}'].add(f'subtask_{idx2}')
                        else:
                            for sub_item in item:
                                if f'subtask_{merged_chain[idx2]}' in cluster_to_dependencies[str(sub_item)[-1]][0]:
                                    mas_dependencies[f'subtask_{idx1}'].add(f'subtask_{idx2}')     
                                    break  
                else:
                    for idx2 in range(idx1 + 1, len(merged_chain)):
                        if isinstance(merged_chain[idx2], list):
                            for item2 in merged_chain[idx2]:
                                if f'subtask_{item2}' in cluster_to_dependencies[str(item)[-1]][0]:
                                    mas_dependencies[f'subtask_{idx1}'].add(f'subtask_{idx2}')
                                    break
                        else:
                            print(f'subtask_{merged_chain[idx2]}', cluster_to_dependencies[str(item)[-1]][0])
                            if f'subtask_{merged_chain[idx2]}' in cluster_to_dependencies[str(item)[-1]][0]:
                                mas_dependencies[f'subtask_{idx1}'].add(f'subtask_{idx2}')
                        
            for idx1, subtask in enumerate(merged_workflow):
                abstract_subtask_list.append({
                    'subtask_id': f'subtask_{idx1}',
                    'subtask_name': subtask['subtask_name'],
                    'abstracted_objective': subtask['abstracted_objective'],
                    'agent_collaboration': subtask['agent_collaboration'],
                    'dependencies': list(mas_dependencies[f"subtask_{idx1}"])
                })
                
            abstract_subtask_list_desc = {}
            for id_, subtask in enumerate(abstract_subtask_list):
                abstract_subtask_list_desc[f"Stage {id_}"] = {
                    "Title": subtask['subtask_name'],
                    'Objectives': subtask['abstracted_objective']
                }
            
            abstract_workflow_description.append({
                'name': f"abstracted_workflow_{mas_idx}",
                'flow': abstract_subtask_list_desc,
                'code_path': f'{workflow_path}/abstracted_workflow/abstracted_workflow_desc_{mas_idx}.json',
                'chain': merged_chain
            })
            with open(f'{workflow_path}/abstracted_workflow/abstract_workflow_description.json', 'w', encoding='utf-8') as f:
                json.dump(abstract_workflow_description, f, ensure_ascii=False, indent=4)
                
            with open(f'{workflow_path}/abstracted_workflow/abstracted_workflow_desc_{mas_idx}.json', 'w', encoding='utf-8') as f:
                json.dump(abstract_subtask_list, f, ensure_ascii=False, indent=4)
            
            with open(f'{workflow_path}/abstracted_workflow/workflow_chains.json', 'w', encoding='utf-8') as f:
                json.dump(merged_workflow_chains, f, ensure_ascii=False, indent=4)
            
            mas_idx += 1
            # break

    async def __call__(self,  query: str, mas: str):
        subtasks = await self.analyze(query, mas)
        
        # subtasks = [{'subtask_id': 'subtask_1', 'objective': "Calculate Aya's walking speed s based on the information that when she walks at s km/h, the walk takes 4 hours including t minutes spent in the coffee shop.", 'supporting_info': 'Aya walks 9 kilometers at speed s, taking a total of 4 hours, which includes t minutes spent in the coffee shop.', 'agent_collaboration': 'CoT', 'dependencies': []}, {'subtask_id': 'subtask_2', 'objective': 'Determine the time spent in the coffee shop t using the information that when she walks at s+2 km/h, the walk takes 2 hours and 24 minutes including t minutes in the coffee shop.', 'supporting_info': 'The output from Sub-task 1 provides the value of s, which is necessary to calculate t. The total time for the walk at s+2 km/h is 2 hours and 24 minutes.', 'agent_collaboration': 'SC_CoT', 'dependencies': ['subtask_1']}, {'subtask_id': 'subtask_3', 'objective': 'Calculate the time it takes for Aya to walk 9 km at s+1/2 km/h, including the t minutes spent in the coffee shop.', 'supporting_info': 'The outputs from Sub-task 1 and Sub-task 2 provide the values of s and t, which are necessary to calculate the total time for the walk at the adjusted speed.', 'agent_collaboration': 'Reflexion', 'dependencies': ['subtask_1', 'subtask_2']}]
        
        abstracted_subtasks = await self.abstract_task_decomposition(query, subtasks)
        
        for idx, subtask in enumerate(subtasks):
            abstracted_subtask = abstracted_subtasks[idx]
            if subtask['subtask_id'] == abstracted_subtask['subtask_id']:
                subtasks[idx]['abstracted_objective'] = abstracted_subtask['abstracted_objective']
                subtasks[idx]['subtask_name'] = abstracted_subtask['subtask_name']
        
        return subtasks
        
    
async def main():
    abstractor = MASAbstraction()    
    print("Init Abstractor successfully")
    
    print("Start time")
    start_time = time.time()
    
    mas_zero_workflow = []
    
    with open("mas_zero_gpqa_diamond.json", "r", encoding="utf-8") as f:
        mas_zero_workflow = json.load(f)
    mas_zero_workflow = mas_zero_workflow[:750]
    print(len(mas_zero_workflow))
    
    # with open("mas_zero_aime24.json", "r", encoding="utf-8") as f:
    #     mas_zero_workflow = json.load(f)
        
    print("============= Read data successfully ==============")
    
    for idx, mas in enumerate(mas_zero_workflow):
        mas_idx = int(idx / 5)
        iteration = int(mas['iteration'])
        
        if os.path.isfile(f"workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa_diamond_test/mas_zero_workflow_analysis_{mas_idx}_iteration_{iteration}.json"):
            continue 
        
        if iteration != 0:
            continue
        
        print(f"Workflow {mas_idx}, iteration: {iteration}")
        subtask_list = await abstractor(mas['problem'], mas['code'])
    
        dir_path = f"workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa_diamond_test"
        file_path = f"{dir_path}/mas_zero_workflow_analysis_{mas_idx}_iteration_{iteration}.json"

        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)

        # Write the JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(subtask_list, f, ensure_ascii=False, indent=4)
            
    cluster_to_subtask, subtask_to_cluster, kmeans, pca, abstracted_subtasks_list, mas_chain = await abstractor.clustering_subtasks_list("workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa_diamond_test")
    cluster_to_agent_collaboration = {str(idx): subtask.agent_collaboration for idx, subtask in enumerate(abstracted_subtasks_list)}
    cluster_to_subtask_name = {str(idx): subtask.name for idx, subtask in enumerate(abstracted_subtasks_list)}
    cluster_to_dependencies = {str(idx): subtask.dependencies for idx, subtask in enumerate(abstracted_subtasks_list)}
    await abstractor.clustering_workflow("workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa_diamond_test", cluster_to_subtask, cluster_to_agent_collaboration, cluster_to_subtask_name, cluster_to_dependencies, mas_chain)
    
    end_time = time.time()
    
    print("Total time: ", end_time - start_time)
    print("Total cost: ", get_global("global_COST_TOTAL"))
    
if __name__ == "__main__":
    model_sampler_map = {
        "o4-mini": OChatCompletionSampler(
            model="o4-mini",
        ),
        "o3-mini": OChatCompletionSampler(
            model="o3-mini",
        ),
        "gpt-4o_chatgpt": ChatCompletionSampler(
            model="gpt-4o",
        ),
        "text-embedding-3-large": ChatCompletionSampler(
            model="text-embedding-3-large",
        ),
        "gpt-4.1-mini": ChatCompletionSampler(
            model="gpt-4.1-mini",
        ),
        "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
            model="gpt-4o-mini",
        )
    }
    set_global("global_model_sampler_map", model_sampler_map)
    set_global("global_COST_TOTAL", 0.0)
    
    asyncio.run(main())
    
    print("Total cost: ", get_global("global_COST_TOTAL"))