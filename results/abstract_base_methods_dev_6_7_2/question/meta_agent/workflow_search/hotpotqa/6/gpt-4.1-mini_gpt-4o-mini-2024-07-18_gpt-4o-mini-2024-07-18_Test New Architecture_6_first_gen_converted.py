async def forward_6(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start sequential
    
    # Stage 0: generate candidate outputs
    # Subtask 1: Identify the name of Jaclyn Stapp's husband using CoT with self-consistency
    cot_sc_instruction1 = "Subtask 1: Identify the name of Jaclyn Stapp's husband based on the query context."
    N = self.max_sc
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc1,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, identifying Jaclyn Stapp's husband, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Subtask 2: Identify the band for which Jaclyn Stapp's husband was the frontman using CoT with self-consistency
    cot_sc_instruction2 = "Subtask 2: Identify the band for which Jaclyn Stapp's husband was the frontman, based on the answer from Subtask 1."
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
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, identifying band fronted by husband, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    # Subtask 3: Determine the year the band disbanded using CoT with self-consistency
    cot_sc_instruction3 = "Subtask 3: Determine the year the band disbanded, based on the answer from Subtask 2."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, determining band disband year, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Control Flow 1: start loop - loop over subtasks 1 to 3 outputs for aggregation
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Subtask 4: Consolidate the identified husband, band, and disband year information into a coherent answer."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results1['answer'], results2['answer'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "answers from subtasks 1, 2, 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating husband, band, and disband year, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    # Stage 1: validate consolidated output
    review_instruction = "Subtask 5: Validate the consolidated answer for accuracy and completeness."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing consolidated answer, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
