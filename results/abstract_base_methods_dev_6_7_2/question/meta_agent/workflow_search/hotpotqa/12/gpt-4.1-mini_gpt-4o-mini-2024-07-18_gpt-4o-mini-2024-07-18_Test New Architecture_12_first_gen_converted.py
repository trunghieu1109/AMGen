async def forward_12(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Identify the current head of the Foreign Relations Department of the Rastriya Janashakti Party"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying head of Foreign Relations Department, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Sub-task 2: List all degree abbreviations (MS, M.S., ScM) and their possible meanings in academic fields"
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, listing degree abbreviations and meanings, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    degree_abbreviations = ["MS", "M.S.", "ScM"]
    candidate_fields_all = []

    for idx, degree_abbr in enumerate(degree_abbreviations, start=3):
        cot_sc_instruction = f"Sub-task {idx}: Generate candidate fields of study associated with the degree abbreviation {degree_abbr} held by the identified person"
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], degree_abbr],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", f"degree abbreviation {degree_abbr}"]
        }
        results_sc = await self.sc_cot(
            subtask_id=f"subtask_{idx}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=self.max_sc
        )
        for i in range(self.max_sc):
            agents.append(f"CoT-SC agent {results_sc['cot_agent'][i].id}, candidate fields for {degree_abbr}, thinking: {results_sc['list_thinking'][i]}; answer: {results_sc['list_answer'][i]}")
        sub_tasks.append(f"Sub-task {idx} output: thinking - {results_sc['thinking'].content}; answer - {results_sc['answer'].content}")
        logs.append(results_sc['subtask_desc'])
        candidate_fields_all.append(results_sc['answer'].content)

    aggregate_instruction = "Sub-task 6: Consolidate multiple candidate fields of study into a single coherent field based on consistency and relevance"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_fields_all,
        'temperature': 0.0,
        'context': ["user query", "candidate fields from subtask 3 to subtask 5"]
    }
    results6 = await self.aggregate(
        subtask_id="subtask_6",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results6['aggregate_agent'].id}, consolidating candidate fields, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    review_instruction = "Sub-task 7: Validate the consolidated field of study against reliable sources to confirm accuracy and completeness"
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results7['review_agent'].id}, validating consolidated field, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
