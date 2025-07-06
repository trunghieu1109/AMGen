async def forward_26(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Stage 1: Iterative Quality Enhancement (Reflexion | Revise) in a loop
    loop_iterations = 3
    reflexion_results = []
    for i in range(loop_iterations):
        revise_instruction = f"Subtask {i+1}: Iteratively evaluate and modify the initial reasoning about the number of field goals Gostkowski kicked in the first quarter to improve clarity, consistency, and completeness. Iteration {i+1}."
        revise_desc = {
            'instruction': revise_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results = await self.reflexion(
            subtask_id=f"subtask_{i+1}",
            cot_reflect_desc=revise_desc,
            critic_desc=revise_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion agent {results['cot_agent'].id}, iteration {i+1}, thinking: {results['list_thinking'][0].content}; answer: {results['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results['list_feedback']))):
            agents.append(f"Critic agent {results['critic_agent'].id}, iteration {i+1}, feedback: {results['list_feedback'][k].content}; correct: {results['list_correct'][k].content}")
            if k + 1 < len(results['list_thinking']) and k + 1 < len(results['list_answer']):
                agents.append(f"Reflexion agent {results['cot_agent'].id}, iteration {i+1}, refining final answer, thinking: {results['list_thinking'][k + 1].content}; answer: {results['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        reflexion_results.append(results)
    
    # Stage 0: Construct Logical Reasoning Sequence (CoT | AnswerGenerate)
    cot_instruction = "Sub-task 4: Decompose the passage information into an ordered logical sequence to determine how many field goals Gostkowski kicked in the first quarter."
    cot_agent_desc = {
        'instruction': cot_instruction,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, decomposing passage, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    # Control Flow 2: end_loop (end of iterative refinement loop)
    
    # Stage 3: Validate Output (Review | Reflexion)
    review_instruction = "Sub-task 5: Evaluate the constructed answer for correctness and reliability based on the passage details."
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
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing answer, feedback: {results5['feedback'].content}; correct: {results5['correct'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results5['feedback'].content}; correct - {results5['correct'].content}")
    logs.append(results5['subtask_desc'])
    
    reflexion_validate_instruction = "Sub-task 6: Based on the review feedback, filter and refine the answer to ensure correctness and reliability."
    reflexion_validate_desc = {
        'instruction': reflexion_validate_instruction,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['feedback']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "feedback of subtask 5"]
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=reflexion_validate_desc,
        critic_desc=reflexion_validate_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion agent {results6['cot_agent'].id}, refining answer after review, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][k].content}; correct: {results6['list_correct'][k].content}")
        if k + 1 < len(results6['list_thinking']) and k + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][k + 1].content}; answer: {results6['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    # Stage 2: Consolidate and select optimal output (Aggregate)
    aggregate_instruction = "Sub-task 7: Aggregate the evaluated outputs and select the most coherent and consistent final answer regarding the number of field goals Gostkowski kicked in the first quarter."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results4['answer'], results6['answer']] + [r['answer'] for r in reflexion_results],
        'temperature': 0.0,
        'context': ["user query", "answers from previous subtasks"]
    }
    results7 = await self.aggregate(
        subtask_id="subtask_7",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results7['aggregate_agent'].id}, aggregating answers, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
