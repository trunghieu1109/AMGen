async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_expansions = []
    cot_instruction = "Sub-task 1: Systematically generate candidate expansions for the new acronym of VIVA Media AG after its 2004 name change by applying structured reasoning and research"
    cot_agent_desc = {
        'instruction': cot_instruction,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    sc_cot_instruction = "Sub-task 2: Based on the initial candidate expansions, consider multiple possible interpretations and variations with self-consistency to generate diverse candidate outputs for the new acronym of VIVA Media AG"
    sc_cot_desc = {
        'instruction': sc_cot_instruction,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    n_candidates = self.max_sc
    for i in range(n_candidates):
        results_cot = await self.cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, generating candidate expansion iteration {i+1}, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
        sc_cot_desc['input'] = [taskInfo, results_cot['thinking'], results_cot['answer']]
        results_sc_cot = await self.sc_cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_sc_desc=sc_cot_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results_sc_cot['cot_agent'][0].id}, refining candidate expansion iteration {i+1}, thinking: {results_sc_cot['list_thinking'][0]}; answer: {results_sc_cot['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 2 iteration {i+1} output: thinking - {results_sc_cot['list_thinking'][0]}; answer - {results_sc_cot['list_answer'][0]}")
        logs.append(results_sc_cot['subtask_desc'])
        candidate_expansions.append(results_sc_cot['list_answer'][0])
    aggregate_instruction = "Sub-task 3: Aggregate multiple candidate acronym expansions generated to produce a single coherent and consistent expansion for the new acronym of VIVA Media AG"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_expansions,
        'temperature': 0.0,
        'context': ["user query", "candidate expansions from previous subtasks"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, integrating candidate expansions, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    review_instruction = "Sub-task 4: Review the consolidated acronym expansion for accuracy, completeness, validity, and correctness"
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results_review = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated expansion, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    cot_instruction_validate = "Sub-task 5: Based on the review feedback, validate and finalize the consolidated acronym expansion for VIVA Media AG's new name"
    cot_agent_desc_validate = {
        'instruction': cot_instruction_validate,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer'], results_review['thinking'], results_review['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "feedback of subtask 4", "correctness of subtask 4"]
    }
    results_validate = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc_validate
    )
    agents.append(f"CoT agent {results_validate['cot_agent'].id}, validating final consolidated expansion, thinking: {results_validate['thinking'].content}; answer: {results_validate['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results_validate['thinking'].content}; answer - {results_validate['answer'].content}")
    logs.append(results_validate['subtask_desc'])
    final_answer = await self.make_final_answer(results_validate['thinking'], results_validate['answer'], sub_tasks, agents)
    return final_answer, logs