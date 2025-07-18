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
from utils import  extract_xml
from shared_vars import set_global, get_global
from pathlib import Path
import asyncio
from llm_agent_base import LLMAgentBase

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
            "context": cot_agent_desc.get('context_desc', ['user query']),
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
                                model=self.node_model, temperature=0.5) for _ in range(n_repeat)]
        list_thinking = []
        list_answer = []
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_agent_desc.get('instruction', "Please do step-by-step"),
            "final_decision_instruction": cot_agent_desc.get("final_decision_instruction", f"Synthesize and choose the most consistent answer for this problem: {cot_agent_desc.get('instruction', "Please do step-by-step")}"),
            "context": cot_agent_desc.get('context_desc', ['user query']),
            "agent_collaboration": "SC_CoT"
        }
        
        # print("Node model: ", self.node_model)
        
        # print("CoT prompt: ", subtask_desc['instruction'])
        # print("Final Decision prompt: ", subtask_desc['final_decision_instruction'])
        # print("n_repeat ", n_repeat)
        
        # print("\n============ Input of Self Consistency CoT ============\n", cot_agent_desc.get('input', []))
        
        # for k, v in subtask_desc.items():
        #     print("key: ", k)
        #     print('\n')
        #     print(v)
        #     print('\n')
        for i in range(n_repeat):
            # Each CoT-SC agent tries to calculate all possible cases independently
            thinking, answer = await cot_agents[i](cot_agent_desc.get('input', []), subtask_desc['instruction'], is_sub_task=True)
            list_thinking.append(thinking)
            list_answer.append(answer)
            # print(f"Attempt {i}: ", subtask_desc['instruction'])
            # print(f"Attempt {i}: ", answer.content)
            # print(cot_agent_desc.get('input', []))
            # print(subtask_desc['instruction'])
            
        # The most common answer is chosen for consistency and accuracy.
        final_decision_agent = LLMAgentBase(['thinking', 'answer'], "Final Decision Agent", 
                                            model=self.node_model, temperature=0.0)
        thinking, answer = await final_decision_agent(cot_agent_desc.get('input', []) + list_thinking + list_answer, 
                                                    subtask_desc['final_decision_instruction'], 
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
            'subtask_desc': subtask_desc
        }, subtask_desc
    
    async def reflexion(self, subtask_id, reflect_desc, n_repeat = 1):
        reflect_inst =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct"
        
        cot_agent = LLMAgentBase(['thinking', 'answer'], "Chain-of-Thought Agent", model=self.node_model, temperature=reflect_desc.get('temperature', 0.0))
        critic_agent = LLMAgentBase(['feedback', 'correct'], "Critic Agent", model=self.node_model, temperature=0.0)
        
        # Input for CoT agent
        cot_inputs = reflect_desc.get('input', [])
        
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": reflect_desc.get('instruction', "Please do step-by-step") + reflect_inst,
            "critic_instruction": reflect_desc.get('critic_instruction', f"Make review about solution of this problem: {reflect_desc.get('instruction', "Please do step-by-step") + reflect_inst}") + critic_inst,
            "context": reflect_desc.get('context_desc', ['user query']),
            "agent_collaboration": "Reflexion"
        }
        
        # feedbacks, corrects = [], []
        # thinkings, answers = [], []
        
        # Generate the first version
        thinking, answer = await cot_agent(cot_inputs, subtask_desc['instruction'], 0, is_sub_task=True)
        # thinkings.append(thinking)
        # answers.append(answer)
        for i in range(n_repeat):
            # Critic agent debates and criticizes pros and cons of previous version
            feedback, correct = await critic_agent(reflect_desc.get('input', []) + [thinking, answer], 
                                        subtask_desc['critic_instruction'], i, is_sub_task=True)
            # feedbacks.append(feedback)
            # corrects.append(correct)
            if correct.content == "True":
                break
            
            cot_inputs.extend([thinking, answer, feedback])
            thinking, answer = await cot_agent(cot_inputs, subtask_desc['instruction'], i + 1, is_sub_task=True)
            # thinkings.append(thinking)
            # answers.append(answer)
            
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
        }, subtask_desc
    
    async def debate(self, subtask_id, debate_desc, n_repeat = 1):
        debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
        final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
        debate_agents = [LLMAgentBase(['thinking', 'answer'], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=debate_desc.get('temperature', 0.5)) 
                        for role in self.debate_role]

        all_thinking = [[] for _ in range(n_repeat)]
        all_answer = [[] for _ in range(n_repeat)]
        
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": debate_desc.get('instruction', "Please do step-by-step") + debate_instr,
            "final_decision_instruction": debate_desc.get("final_decision_instruction", "Make final answer for the problem") + final_instr,
            "context": debate_desc.get('context_desc', ['user query']),
            "agent_collaboration": "Debate"
        }
        
        for r in range(n_repeat):
            # N_max_5 rounds of debating
            for i, agent in enumerate(debate_agents):
                # Each agent proposes its solution
                if r == 0:
                    thinking, answer = await agent(debate_desc.get('input', []), 
                                            subtask_desc['instruction'], r, is_sub_task=True)
                else:
                    # Generate next solution based on comments and counter-arguments from other debaters
                    input_infos = debate_desc.get('input', []) + all_thinking[r-1] + all_answer[r-1]
                    thinking, answer = await agent(input_infos, subtask_desc['instruction'], r, is_sub_task=True)
                
                all_thinking[r].append(thinking)
                all_answer[r].append(answer)
        
        # Final decision agent makes final decision
        final_decision_agent = LLMAgentBase(['thinking', 'answer'], "Final Decision Agent", 
                                            model=self.node_model, temperature=0.0)
        thinking, answer = await final_decision_agent(debate_desc.get('input', []) + all_thinking[-1] + all_answer[-1], 
                                                    debate_desc.get('instruction', "Please do step-by-step"), 
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
