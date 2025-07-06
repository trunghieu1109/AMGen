async def forward_8(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Identify the Dutch manager Ronald Koeman's predecessor as Southampton F.C. manager in the 2014–15 season"
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc_1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying predecessor of Ronald Koeman, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Determine the Argentine former footballer who was replaced by Ronald Koeman in the 2014–15 Southampton F.C. season, based on predecessor identified in Subtask 1"
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc_2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, determining Argentine former footballer replaced by Koeman, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = "Subtask 3: Generate candidate birth dates for the identified Argentine former footballer using self-consistency chain-of-thought reasoning"
    N = self.max_sc
    cot_sc_desc_3 = {
        'instruction': cot_sc_instruction_3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc_3,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, generating candidate birth date #{idx+1}, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    aggregate_instruction_4 = "Subtask 4: Consolidate multiple candidate birth dates into a single coherent birth date"
    aggregate_desc_4 = {
        'instruction': aggregate_instruction_4,
        'input': [taskInfo] + results3['list_answer'],
        'temperature': 0.0,
        'context': ["user query", "candidate birth dates from subtask 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc_4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate birth dates, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    review_instruction_5 = "Subtask 5: Validate the consolidated birth date for accuracy and completeness"
    review_desc_5 = {
        'instruction': review_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc_5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, validating consolidated birth date, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
