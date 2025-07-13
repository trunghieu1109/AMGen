import argparse
from datasets import load_dataset
import pandas as pd
from common import HTML_JINJA, SingleEvalResult
import search
import apply_abstract_workflow_v5
import re
import os
import common
from common import ANSWER_PATTERN, check_equality
from sampler.chat_completion_sampler import ChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.together_completion_sampler import ChatCompletionSampler as ToChatCompletionSampler
from sampler.vllm_completion_sampler import ChatCompletionSampler as VllmChatCompletionSampler
import json
from utils import load_questions, load_questions_drop, load_questions_gsm8k, load_questions_hotpotqa
from prompts.swe.patch_oracle import AGENTLESS_REPAIR
from swe_utils import run_swebench_evaluation, sanity_check
from utils import  extract_xml
from shared_vars import set_global, get_global
from pathlib import Path
from sklearn.model_selection import train_test_split
import asyncio
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--valid_size', type=int, default=128)
parser.add_argument('--test_size', type=int, default=800)
parser.add_argument('--shuffle_seed', type=int, default=0)
parser.add_argument('--n_repeats', type=int, default=1)
parser.add_argument('--multiprocessing', action='store_true', default=True)
parser.add_argument('--max_workers', type=int, default=48)
parser.add_argument('--debug', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default='results/')
parser.add_argument('--expr_name', type=str)
parser.add_argument('--n_generation', type=int, default=1)
parser.add_argument('--max_round', type=int, default=1)
parser.add_argument('--max_sc', type=int, default=3)
parser.add_argument('--debug_max', type=int, default=3)
parser.add_argument('--option', type=str, default='')
parser.add_argument('--meta_model',
                    type=str)
parser.add_argument('--node_model',
                    type=str)
parser.add_argument('--verifier_model',
                    type=str,
                    default="o4-mini")
                    # gpt-4o
parser.add_argument('--shorten_context', action='store_true')
parser.add_argument('--merge_context', action='store_true')

                    
parser.add_argument(
    "--blocks", type=str, nargs="*", help="Number of examples to use (overrides default)"
)
parser.add_argument('--dataset', type=str)
parser.add_argument(
    "--given_examples", type=int, nargs="*", help="Number of examples to use (overrides default)"
)
parser.add_argument(
    "--use_oracle_verifier", action='store_true'
)
parser.add_argument(
    "--defer_verifier", action='store_true'
)
parser.add_argument(
    "--no_decompose", action='store_true'
)
parser.add_argument(
    "--no_meta_reward", action='store_true'
)

class DataScorer:

    def __init__(self, dataset, technique):
        self.dataset = dataset
        self.technique = technique
        self.equality_checker = ChatCompletionSampler(model="gpt-4o-mini-2024-07-18")
        self.LETTER_TO_INDEX = {'A': 1, 'B': 2, 'C': 3, 'D': 4}


    async def run_score(self, question, answer, extracted_answer, use_oracle_verifier, judge_path, instance_id, n, code_snippet):

        if 'swe_bench' in self.dataset:
            score, percentage, passed_tests, total_tests = run_swebench_evaluation(judge_path, instance_id, extracted_answer, self.technique, n, code_snippet)

            with open(judge_path, 'a+') as judge_file:
                judge_file.write(f'{instance_id} → {passed_tests} passed test | {total_tests} total_tests | {passed_tests}/{total_tests} passed → {percentage:.1f}% | Score: {score}\n')

            return score

        elif 'aime24' in self.dataset:
            quality = await check_equality(self.equality_checker, question, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(quality)
        elif 'drop' in self.dataset:
            quality = await check_equality(self.equality_checker, question, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(quality)
        elif 'hotpotqa' in self.dataset:
            quality = await check_equality(self.equality_checker, question, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(quality)
        elif 'gsm8k' in self.dataset:
            quality = await check_equality(self.equality_checker, question, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path)
            return float(quality)
        elif 'gpqa_diamond' in self.dataset:
            
            res = extracted_answer
            is_early_stop = False
            try:
                if isinstance(res, str) and res in self.LETTER_TO_INDEX:
                    predicted_idx = self.LETTER_TO_INDEX[res]
                elif 'A)' in res or res == 'A':
                    predicted_idx = 1
                elif 'B)' in res or res == 'B':
                    predicted_idx = 2
                elif 'C)' in res or res == 'C':
                    predicted_idx = 3
                elif 'D)' in res or res == 'D':
                    predicted_idx = 4
                elif isinstance(res, list):
                    try_res = res[1]
                    predicted_idx = self.LETTER_TO_INDEX[try_res.content]
                elif res.content in self.LETTER_TO_INDEX:
                    predicted_idx = self.LETTER_TO_INDEX[res.content]
                elif 'A)' in res.content or res == 'A':
                    predicted_idx = 1
                elif 'B)' in res.content or res == 'B':
                    predicted_idx = 2
                elif 'C)' in res.content or res == 'C':
                    predicted_idx = 3
                elif 'D)' in res.content or res == 'D':
                    predicted_idx = 4
                else:
                    print(f"error in q {q_idx}")
                    score = 0
                    is_early_stop = True
            except Exception as e:
                score = 0
                is_early_stop = True

            if not is_early_stop: # if cannot find predicted_idx, then done
                if predicted_idx == answer:
                    score = 1
                else:
                    score = 0
                    
            # print(f"predicted_idx: {predicted_idx}")

            print(f'extracted_answer: {extracted_answer}; answer: {answer}; score: {score}')

            return score
        
        else:
            raise NotImplementedError

    async def score(self, example_id, n, prompt_message, question, response_text, answer, sub_tasks_text, use_oracle_verifier, judge_path, response_path, response_dict, instance_id, code_snippet):

        if 'swe_bench' in self.dataset:
            extracted_answer = response_text.split('\n\nAnswer:', 1)[-1].strip()
            if '<patch>' in extracted_answer:
                extracted_answer = extract_xml(extracted_answer, 'patch').strip()   
        else:
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None    
            extracted_answer = extracted_answer.strip()


        print('extracted_answer: ',extracted_answer)

        with open(judge_path, 'a+') as judge_file:
            judge_file.write(f'Question: {question}\nIteration: {n}\nproposed answer: {response_text}\nExtracted answer: {extracted_answer}\nCorrect answer: {answer}\n')

        if use_oracle_verifier:
            score_oracle_verifier = await self.run_score(question, answer, extracted_answer, use_oracle_verifier=True, judge_path=judge_path, instance_id=instance_id, n=n, code_snippet=code_snippet)
            score = score_oracle_verifier
            score_model_verifier = None 
        else:
            if sub_tasks_text is None:
                score_model_verifier = await self.run_score(mode_verifier, question, response_text, use_oracle_verifier=False, judge_path=judge_path, instance_id=instance_id, n=n, code_snippet=code_snippet)
            else:
                score_model_verifier = await self.run_score(mode_verifier, question, sub_tasks_text, use_oracle_verifier=False, judge_path=judge_path, instance_id=instance_id, n=n, code_snippet=code_snippet)
            score = score_model_verifier

        html = common.jinja_env.from_string(HTML_JINJA).render(
            prompt_messages=prompt_message,
            next_message=dict(content=response_text, role="assistant"),
            score=score,
            correct_answer=answer,
            extracted_answer=extracted_answer,
        )
        convo = prompt_message + [dict(content=response_text, role="assistant")]
        results = SingleEvalResult(html=html, score=score, convo=convo)
        return score_oracle_verifier, score_model_verifier, results

def split_array(arr, test_size):
    # val_set, test_set = train_test_split(arr, test_size=test_size, random_state=42)
    test_length = int(test_size * len(arr))
    val_length = len(arr) - test_length
    return arr[:val_length], arr[val_length:]

def transform_code(code_str):
    mapping = {
        r'\bglobal_max_sc\b': 'self.max_sc',
        r'\bglobal_max_round\b': 'self.max_round',
        r'\bglobal_debate_role\b': 'self.debate_role',
        r'\bglobal_node_model\b': 'self.node_model',
        r'\bglobal_cot_instruction\b': 'self.cot_instruction'
    }
    for pattern, replacement in mapping.items():
        code_str = re.sub(pattern, replacement, code_str)
    return code_str

def fix_sync_func(code_str):
    
    pattern = r'(\b(think[\w]*|t[\w])\s*,\s*(answer[\w]*|a[\w])\s*=\s*)(?!await\s)'
    replacement = r'\1await '

    pattern_output = re.compile(r'(\b\w*outputs\w*\s*=\s*)(?!await\s)')
    pattern_info = re.compile(r'(\b\w*info\w*\s*=\s*)(?!await\s)')

    if not re.compile(pattern_output).search(code_str) and not re.compile(pattern_info).search(code_str):
        converted_code_v1 = re.sub(pattern, replacement, code_str)
        pattern2 = r'(\bfeedback[\w]*\s*,\s*correct[\w]*\s*=\s*)(?!await\s)'
        converted_code_v2 = re.sub(pattern2, replacement, converted_code_v1)
        return converted_code_v2
    else:
        if re.compile(pattern_info).search(code_str):
            converted_code_v1 = re.sub(pattern_info, replacement, code_str)
            return converted_code_v1
        else:
            converted_code_v1 = re.sub(pattern_output, replacement, code_str)
            return converted_code_v1

    return code_str

args = parser.parse_args()

async def run_main():
    
    blocks = args.blocks
    meta_model = args.meta_model
    node_model = args.node_model
    verifier_model = args.verifier_model
    use_oracle_verifier = args.use_oracle_verifier
    max_round = args.max_round
    max_sc = args.max_sc

    SEARCHING_MODE = True

    technique = args.dataset.split('/')[0] 
    data_scorer = DataScorer(args.dataset, technique)

    print('verifier_model: ',verifier_model)
    print('technique: ',technique)
    print('node_model: ',node_model)

    model_sampler_map = {
        "o4-mini": OChatCompletionSampler(
            model="o4-mini",
        ),
        "o3-mini": OChatCompletionSampler(
            model="o3-mini",
        ),
        "o1": OChatCompletionSampler(
            model="o1",
        ),
        "gpt-4o_chatgpt": ChatCompletionSampler(
            model="gpt-4o",
        ),
        "gpt-4.1-mini": ChatCompletionSampler(
            model="gpt-4.1-mini",
        ),
        "gpt-4.1-nano": ChatCompletionSampler(
            model="gpt-4.1-nano",
        ),
        "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
            model="gpt-4o-mini",
        ),
        "text-embedding-3-large": ChatCompletionSampler(
            model="text-embedding-3-large",
        ),
        "qwen-2.5-32b-instr": VllmChatCompletionSampler(
            model="qwen-2.5-32b-instr",
        ),
        "qwq-32b": ToChatCompletionSampler(
            model="Qwen/Qwen2.5-32B-Instruct",
        ),
        "llama-3.3-70b-instr": ToChatCompletionSampler(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
    }

    json_model = ['gpt', 'o']
    xml_model = ['qwen', 'llama-3.3']


    if any(kw in node_model for kw in json_model):

        FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n\n"""
        set_global("global_format_choice", 'json')

    elif any(kw in node_model for kw in xml_model):
        FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following XML format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n"""
        set_global("global_format_choice", 'xml')

    else:
        raise NotImplementedError


    mode_verifier = model_sampler_map[verifier_model]

    set_global("global_FORMAT_INST", FORMAT_INST)
    set_global("global_model_sampler_map", model_sampler_map)
    set_global("global_shorten_context", args.shorten_context)
    set_global("global_merge_context", args.merge_context)
    set_global("global_COST_TOTAL", 0.0)
    set_global("global_COST_EXECUTION", 0.0)
    set_global("global_no_decompose", args.no_decompose)
    set_global("global_no_meta_reward", args.no_meta_reward)

    print('shorten_context: ',args.shorten_context)
    print('merge_context: ',args.merge_context)
    print('global_no_meta_reward: ',args.no_meta_reward)
    print('global_no_decompose: ',args.no_decompose)
    
    # load abstract workflow
    aw_desc_path = 'workflow_analysis-gpt-4o-mini-o4-mini_v8-gpqa-diamond_v3/abstracted_workflow/abstract_workflow_description.json'
    abstract_workflow = []
    
    with open(aw_desc_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        code_path = item.get('code_path')   
        if code_path and os.path.isfile(code_path):
            with open(code_path, 'r', encoding='utf-8') as code_file:
                code_content = code_file.read()
                abstract_workflow.append({
                    'name': item['name'],
                    'flow': item['flow'],
                    'code': code_content,
                    'chain': item['chain'],
                    'code_path': item['code_path']
                })
                # print(abstract_workflow[-1])
        else:
            print(f'Không tìm thấy file tại đường dẫn: {code_path}')
            
    # abstract_workflow = [abstract_workflow[0]]
    converted_mas_zero_workflow = None
    cot_instruction = ""
    output_description = ""
    debate_role = []
    examples = []
    patterns = ['cot', 'sc_cot', 'reflexion', 'debate']

    if 'aime24' in args.dataset:
        # load mas_zero workflow
        with open('mas_zero_aime24.json', 'r') as file:
            mas_zero_workflow = json.load(file)

        converted_mas_zero_workflow = []
        for idx, workflow in enumerate(mas_zero_workflow):
            converted_workflow = workflow
            converted_workflow['code'] = transform_code(workflow['code'])
            if int(idx / 5) == 9:
                converted_mas_zero_workflow.append(converted_workflow)
                continue
            converted_workflow['code'] = fix_sync_func(converted_workflow['code'])
            converted_mas_zero_workflow.append(converted_workflow)
            
        print("Total mas zero workflow: ", len(converted_mas_zero_workflow))

        cot_instruction = "Please think step by step and then solve the task."
        # output_description = "Return ONLY an integer. DO NOT return anything other than the integer answer."
        output_description = "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."

        debate_role = ['Algebraist', 'Geometer', 'Logician', 'The Universal Mathematician']

        dataset = load_dataset("simplescaling/aime24_nofigures")
        df = pd.DataFrame(dataset['train'])
        examples = [row.to_dict() for _, row in df.iterrows()]
        # examples = [examples[28]]
        # examples = examples[:11]
        test_size = 0.6
        
        val_set, test_set = split_array(examples, test_size)
        
        print("Validation Set Length: ", len(val_set))
        print("Test Set Length: ", len(test_set))
        
        set_global("global_output_description", output_description)
        set_global("global_score_compute", data_scorer.score)
        set_global("global_max_round", max_round)
        set_global("global_max_sc", max_sc)
        set_global("global_debate_role", debate_role)
        set_global("global_cot_instruction", cot_instruction)
        set_global("global_node_model", node_model)
        set_global("global_use_oracle_verifier", use_oracle_verifier)
        set_global("global_dataset", args.dataset)
        
    elif 'drop' in args.dataset:

        cot_instruction = "Please think step by step and then solve the task."
        # output_description = "Return ONLY the alphabet choice, i.e. A or B or C or D."
        output_description = "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."
        # need to consider sub-task output as well (no fixed form for sub-tasks)
        debate_role = ['Math Professor', 'Grade School Teacher']
        
        questions = load_questions_drop('dataset/drop_validate.csv', seed=0)

        examples = [{'problem': questions[i].question, 'answer': questions[i].answer} for i in range(len(questions))]
        examples = examples[:50]
        # examples = [examples[0]]
        set_global("global_output_description", output_description)
        set_global("global_score_compute", data_scorer.score)
        set_global("global_max_round", max_round)
        set_global("global_max_sc", max_sc)
        set_global("global_debate_role", debate_role)
        set_global("global_cot_instruction", cot_instruction)
        set_global("global_node_model", node_model)
        set_global("global_use_oracle_verifier", use_oracle_verifier)
        set_global("global_dataset", args.dataset)
    elif 'gsm8k' in args.dataset:

        cot_instruction = "Please think step by step and then solve the task."
        # output_description = "Return ONLY the alphabet choice, i.e. A or B or C or D."
        output_description = "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."
        # need to consider sub-task output as well (no fixed form for sub-tasks)
        debate_role = ['Math Professor', 'Grade School Teacher']
        
        questions = load_questions_gsm8k('dataset/gsm8k_validate.csv', seed=0)

        examples = [{'problem': questions[i].question, 'answer': questions[i].answer} for i in range(len(questions))]
        examples = examples[:20]
        # examples = [examples[0]]
        set_global("global_output_description", output_description)
        set_global("global_score_compute", data_scorer.score)
        set_global("global_max_round", max_round)
        set_global("global_max_sc", max_sc)
        set_global("global_debate_role", debate_role)
        set_global("global_cot_instruction", cot_instruction)
        set_global("global_node_model", node_model)
        set_global("global_use_oracle_verifier", use_oracle_verifier)
        set_global("global_dataset", args.dataset)
    elif 'hotpotqa' in args.dataset:

        cot_instruction = "Please think step by step and then solve the task."
        # output_description = "Return ONLY the alphabet choice, i.e. A or B or C or D."
        output_description = "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."
        # need to consider sub-task output as well (no fixed form for sub-tasks)
        debate_role = ['Math Professor', 'Grade School Teacher', 'Consultant', 'Science Generalist']
        
        questions = load_questions_gsm8k('dataset/hotpotqa_validate.csv', seed=0)

        examples = [{'problem': questions[i].question, 'answer': questions[i].answer} for i in range(len(questions))]
        examples = examples[:10]
        # examples = [examples[0]]
        set_global("global_output_description", output_description)
        set_global("global_score_compute", data_scorer.score)
        set_global("global_max_round", max_round)
        set_global("global_max_sc", max_sc)
        set_global("global_debate_role", debate_role)
        set_global("global_cot_instruction", cot_instruction)
        set_global("global_node_model", node_model)
        set_global("global_use_oracle_verifier", use_oracle_verifier)
        set_global("global_dataset", args.dataset)
    else:
        # load mas_zero workflow
        with open('mas_zero_gpqa_diamond.json', 'r') as file:
            mas_zero_workflow = json.load(file)

        converted_mas_zero_workflow = []
        for idx, workflow in enumerate(mas_zero_workflow):
            converted_workflow = workflow
            converted_workflow['code'] = transform_code(workflow['code'])
            converted_workflow['code'] = fix_sync_func(converted_workflow['code'])
            converted_mas_zero_workflow.append(converted_workflow)
            
        print("Total mas zero workflow: ", len(converted_mas_zero_workflow))

        cot_instruction = "Please think step by step and then solve the task."
        # output_description = "Return ONLY the alphabet choice, i.e. A or B or C or D."
        output_description = "Return ONLY the alphabet choice, A) or B) or C) or D)."
        # need to consider sub-task output as well (no fixed form for sub-tasks)
        debate_role = ['Biology Expert', 'Physics Expert', 'Chemistry Expert', 'Science Generalist']

        questions = load_questions('dataset/gpqa_diamond.csv', seed=0)
        answers = [question.correct_index for question in questions]

        examples = [{'problem': questions[i], 'answer': answers[i]} for i in range(len(questions))]
        # examples = examples[:170]
        # examples = [examples[170]]
        set_global("global_output_description", output_description)
        set_global("global_score_compute", data_scorer.score)
        set_global("global_max_round", max_round)
        set_global("global_max_sc", max_sc)
        set_global("global_debate_role", debate_role)
        set_global("global_cot_instruction", cot_instruction)
        set_global("global_node_model", node_model)
        set_global("global_use_oracle_verifier", use_oracle_verifier)
        set_global("global_dataset", args.dataset)
    
    final_results = {}
    save_path_ = ""
    
    async def process_example(example_id, example, args, meta_model, verifier_model, abstract_workflow):
        final_results[str(example_id)] = {
            'score': 0,
            'total_time': 0
        }
        instance_id = example_id

        if args.given_examples:
            if example_id not in args.given_examples: return

        args.expr_name = f'dev19_attemp4/question/meta_agent/'
        # args.expr_name = f'abstract_workflow_gpt_4o_chatgpt_o4_mini_v11/question/meta_agent/'
        print('args.expr_name: ', args.expr_name)

        questions = [example['problem']]
        answers = [example['answer']]

        print("Question: ", questions)
        print("Answer: ", answers)

        task_queue = [('task', 'User', q, None, None, None, -1) for q in questions]

        global_questions = get_global("global_questions")
        global_questions[str(example_id)] = questions

        global_answers = get_global("global_answers")
        global_answers[str(example_id)] = answers

        global_response_dict = get_global("global_response_dict")
        global_response_dict[str(example_id)] = []

        set_global("global_answers", global_answers)
        set_global("global_questions", global_questions)
        set_global("global_response_dict", global_response_dict)
        set_global("global_instance_id", instance_id)
        
        score = 0
        total_time = 0
        save_path = ""
        retries = 0
        now = datetime.datetime.now()
        while retries < 2:
            score, total_time, total_execution_time, save_path = await apply_abstract_workflow_v5.apply_abstract_workflow_enhance(args, args.expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow, str(now))
            # score, total_time, total_execution_time, save_path = await apply_abstract_workflow_v5.recheck_mas(args, args.expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow)
            if score > -1:
                break
            retries += 1
        # score, total_time, total_execution_time, save_path = await apply_abstract_workflow_v5.run_single_agent_baselines(args, args.expr_name, example_id, task_queue, meta_model, verifier_model, "reflexion")
        # score, total_time, total_execution_time, save_path = await apply_abstract_workflow_v5.recheck_mas(args, args.expr_name, example_id, task_queue, meta_model, verifier_model, abstract_workflow)
        save_path_ = save_path
        final_results[str(example_id)] = {
            'score': score,
            'total_time': total_time,
            'total_execution_time': total_execution_time
        }
        # await apply_abstract_workflow_v5.test_mas_zero_workflow(args, args.expr_name, example_id, task_queue, meta_model, verifier_model, converted_mas_zero_workflow)
    
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(30)  # Giới hạn 3 task chạy song song

    async def process_example_with_semaphore(example_id, example, args, meta_model, verifier_model, abstract_workflow):
        print(f"\n===================== Run task: {example_id}, Abstract workflow length: {len(abstract_workflow)} =====================\n")
        async with semaphore:
            await process_example(example_id, example, args, meta_model, verifier_model, abstract_workflow)
        
    temp_results = None
            
    # with open("results/workflow_search/gpqa_diamond/mas_zero/gpt-4o_chatgpt_o4-mini_oracle.results", "r", encoding="utf-8") as f:
    #     temp_results = f.read()
    # Tạo task list
    # tasks = [
    #     process_example_with_semaphore(example_id, example, args, meta_model, verifier_model, abstract_workflow)
    #     for example_id, example in enumerate(examples) if f"experiemnt {example_id}:" not in temp_results
    # ]
    
    tasks = [
        process_example_with_semaphore(example_id, example, args, meta_model, verifier_model, abstract_workflow)
        for example_id, example in enumerate(examples)
    ]

    # Chạy tất cả với giới hạn semaphore
    await asyncio.gather(*tasks)
    # print(len(tasks))
    
    total_accuracy = 0
    total_time = 0
    total_cost = get_global("global_COST_TOTAL")
    total_execution_cost = get_global("global_COST_EXECUTION")
    total_execution_time = 0
    for k, v in final_results.items():
        if v['score'] > 0:
            total_accuracy += v['score']
        total_time += v['total_time']
        total_execution_time += v['total_execution_time']
    
    final_results['result'] = {
        'avg_accuracy': total_accuracy / len(examples) * 100,
        'avg_time': total_time / len(examples),
        'avg_cost': total_cost / len(examples),
        'avg_execution_cost': total_execution_cost / len(examples),
        'avg_execution_time': total_execution_time / len(examples),
    }
    
    print("Total time: ", total_time)
    print("Total execution time: ", total_execution_time)
    print("Total cost: ", total_cost)
    print("Total execution cost: ", total_execution_cost)
    print("Total acc: ", total_accuracy)
    
    print("Final results: ", final_results['result'] )
    
    print("Time ratio: ", final_results['result']['avg_execution_time'] / final_results['result']['avg_time'])
    print("Execution Cost ratio: ", final_results['result']['avg_execution_cost'] / final_results['result']['avg_cost'])
    
    end_time = time.time()
    print("Total time: ", end_time - start_time)
    
asyncio.run(run_main())