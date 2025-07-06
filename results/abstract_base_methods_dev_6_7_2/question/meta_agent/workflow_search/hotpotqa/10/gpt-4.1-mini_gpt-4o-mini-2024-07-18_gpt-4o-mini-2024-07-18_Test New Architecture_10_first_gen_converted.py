async def forward_10(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Identify the Vice Presidential candidate associated with the 'Stronger Together' campaign."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying Vice Presidential candidate, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Determine if the identified Vice Presidential candidate was a Senator, considering multiple possible interpretations for accuracy."
    N = self.max_sc
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, checking Senator status, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    candidate_was_senator = False
    if 'yes' in results2['answer'].content.lower() or 'senator' in results2['answer'].content.lower():
        candidate_was_senator = True

    candidate_states = []
    if candidate_was_senator:
        cot_instruction3 = "Subtask 3: Find the state represented by the Vice Presidential candidate who was a Senator."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, results1['answer'], results2['answer']],
            'temperature': 0.0,
            'context': ["user query", "answer of subtask 1", "answer of subtask 2"]
        }
        results3 = await self.cot(
            subtask_id="subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, finding state represented by Senator candidate, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        candidate_states.append(results3['answer'].content)

    aggregate_instruction4 = "Subtask 4: Consolidate the candidate state outputs to synthesize a single coherent answer."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo] + candidate_states,
        'temperature': 0.0,
        'context': ["user query", "candidate states from subtask 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate states, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    review_instruction5 = "Subtask 5: Validate the consolidated state output for accuracy and completeness."
    review_desc5 = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing consolidated state output, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    if results5['answer'].content.lower() in ['yes', 'correct', 'true']:
        final_thinking = results4['thinking']
        final_answer = results4['answer']
    else:
        final_thinking = results5['thinking']
        final_answer = results5['answer']

    final_answer_processed = await self.make_final_answer(final_thinking, final_answer, sub_tasks, agents)
    return final_answer_processed, logs
