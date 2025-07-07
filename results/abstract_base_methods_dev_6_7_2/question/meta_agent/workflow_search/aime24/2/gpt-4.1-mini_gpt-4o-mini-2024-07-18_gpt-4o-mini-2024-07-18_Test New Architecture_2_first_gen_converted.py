async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Define the sample space including vertices, color assignments, and rotation group for a regular octagon with context from taskInfo"
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, defining sample space, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_sc_instruction2 = "Subtask 2: Identify constraints for rotations mapping blue vertices onto originally red positions with context from taskInfo and output of Subtask 1"
    N = self.max_sc
    cot_sc_desc = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=N
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, identifying constraints, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])
    cot_instruction3 = "Subtask 3: Determine total number of colorings (2^8) and characterize favorable mappings with context from taskInfo and outputs of Subtasks 1 and 2"
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, determining total colorings and favorable mappings, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    favorable_counts = []
    for rotation in range(8):
        cot_instruction4 = f"Subtask {4+rotation}: For rotation {rotation}, count colorings satisfying blue-to-red mapping with context from taskInfo and output of Subtask 3"
        cot_agent_desc4 = {
            'instruction': cot_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results_loop = await self.answer_generate(
            subtask_id=f"subtask_{4+rotation}",
            cot_agent_desc=cot_agent_desc4
        )
        agents.append(f"AnswerGenerate agent {results_loop['cot_agent'].id}, counting colorings for rotation {rotation}, thinking: {results_loop['thinking'].content}; answer: {results_loop['answer'].content}")
        sub_tasks.append(f"Subtask {4+rotation} output: thinking - {results_loop['thinking'].content}; answer - {results_loop['answer'].content}")
        logs.append(results_loop['subtask_desc'])
        favorable_counts.append(results_loop['answer'].content)
    aggregate_instruction = "Subtask 12: Aggregate counts over all rotations to compute total favorable colorings with context from taskInfo and counts from Subtasks 4 to 11"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + favorable_counts,
        'temperature': 0.0,
        'context': ["user query", "counts from subtasks 4 to 11"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_12",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating counts, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Subtask 12 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    programmer_instruction = "Subtask 13: Validate aggregated count and ensure no overcounting of symmetric cases with context from taskInfo and output of Subtask 12"
    programmer_desc = {
        'instruction': programmer_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 12", "answer of subtask 12"]
    }
    results_programmer = await self.programmer(
        subtask_id="subtask_13",
        programmer_desc=programmer_desc
    )
    agents.append(f"Programmer agent {results_programmer['programmer_agent'].id}, validating aggregated count, thinking: {results_programmer['thinking'].content}; answer: {results_programmer['answer'].content}, executing results: {results_programmer['exec_result']}")
    sub_tasks.append(f"Subtask 13 output: thinking - {results_programmer['thinking'].content}; answer - {results_programmer['answer'].content}; output - {results_programmer['exec_result']}")
    logs.append(results_programmer['subtask_desc'])
    condition = True
    if condition:
        cot_instruction_true = "Subtask 14: Identify and clarify that no alternative branch logic is needed for this problem with context from taskInfo and outputs of Subtasks 7 and 8"
        formatter_desc_true = {
            'instruction': cot_instruction_true,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"],
            'format': 'short and concise, without explanation'
        }
        results_true = await self.specific_format(
            subtask_id="subtask_14",
            formatter_desc=formatter_desc_true
        )
        agents.append(f"SpecificFormat agent {results_true['formatter_agent'].id}, clarifying no alternative logic needed, thinking: {results_true['thinking'].content}; answer: {results_true['answer'].content}")
        sub_tasks.append(f"Subtask 14 output: thinking - {results_true['thinking'].content}; answer - {results_true['answer'].content}")
        logs.append(results_true['subtask_desc'])
    else:
        cot_instruction_false = "Subtask 15: (False branch) Placeholder for alternative logic if needed"
        formatter_desc_false = {
            'instruction': cot_instruction_false,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"],
            'format': 'short and concise, without explanation'
        }
        results_false = await self.specific_format(
            subtask_id="subtask_15",
            formatter_desc=formatter_desc_false
        )
        agents.append(f"SpecificFormat agent {results_false['formatter_agent'].id}, false branch logic, thinking: {results_false['thinking'].content}; answer: {results_false['answer'].content}")
        sub_tasks.append(f"Subtask 15 output: thinking - {results_false['thinking'].content}; answer - {results_false['answer'].content}")
        logs.append(results_false['subtask_desc'])
    review_instruction = "Subtask 16: Validate and assess the final probability fraction for correctness and reduced form with context from taskInfo and output of Subtask 8"
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_true['thinking'] if condition else results_false['thinking'], results_true['answer'] if condition else results_false['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 14 or 15", "answer of subtask 14 or 15"]
    }
    results_review = await self.review(
        subtask_id="subtask_16",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, validating final fraction, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Subtask 16 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    format_instruction = "Subtask 17: Enhance clarity of the probability derivation and simplify presentation with context from taskInfo and output of Subtask 9"
    format_desc = {
        'instruction': format_instruction,
        'input': [taskInfo, results_review['thinking'], results_review['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 16", "answer of subtask 16"]
    }
    results_format = await self.specific_format(
        subtask_id="subtask_17",
        formatter_desc=format_desc
    )
    agents.append(f"SpecificFormat agent {results_format['formatter_agent'].id}, enhancing clarity, thinking: {results_format['thinking'].content}; answer: {results_format['answer'].content}")
    sub_tasks.append(f"Subtask 17 output: thinking - {results_format['thinking'].content}; answer - {results_format['answer'].content}")
    logs.append(results_format['subtask_desc'])
    format_final_instruction = "Subtask 18: Format the final answer as m+n after reduction with context from taskInfo"
    format_final_desc = {
        'instruction': format_final_instruction,
        'input': [taskInfo, results_format['thinking'], results_format['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 17", "answer of subtask 17"]
    }
    results_final = await self.specific_format(
        subtask_id="subtask_18",
        formatter_desc=format_final_desc
    )
    agents.append(f"SpecificFormat agent {results_final['formatter_agent'].id}, formatting final answer, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Subtask 18 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
