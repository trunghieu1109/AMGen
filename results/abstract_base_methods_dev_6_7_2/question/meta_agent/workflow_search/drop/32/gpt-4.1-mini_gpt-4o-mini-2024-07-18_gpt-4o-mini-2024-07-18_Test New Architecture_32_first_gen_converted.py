async def forward_32(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose input information into an ordered sequence of logical steps to derive the initial calculation of the time span during which births roughly doubled in Bahrain." 
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing input, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    loop_iterations = 3
    iterative_results = []
    for i in range(loop_iterations):
        revise_instruction2 = f"Subtask 2: Iteratively evaluate and modify the initial calculation and reasoning to enhance clarity, consistency, and completeness of the answer. Iteration {i+1}."
        revise_desc2 = {
            'instruction': revise_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results2 = await self.reflexion(
            subtask_id=f"subtask_2_iter_{i+1}",
            cot_reflect_desc=revise_desc2,
            critic_desc={
                'instruction': "Please review the revised calculation and reasoning for clarity and correctness.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, iteration {i+1}, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results2['list_feedback']))):
            agents.append(f"Critic agent {results2['critic_agent'].id}, iteration {i+1}, feedback: {results2['list_feedback'][k].content}; correct: {results2['list_correct'][k].content}")
            if k + 1 < len(results2['list_thinking']) and k + 1 < len(results2['list_answer']):
                agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, iteration {i+1}, refining answer, thinking: {results2['list_thinking'][k + 1].content}; answer: {results2['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task 2 iteration {i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        iterative_results.append((results2['thinking'], results2['answer']))
    
    # Control Flow 2: end_loop
    
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction3 = "Subtask 3: Aggregate multiple variant outputs from the iterative improvements and select the most coherent and consistent final result for the time span answer."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + [ans for _, ans in iterative_results],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from iterative improvements"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 2: Validate Output (optional)
    review_instruction4 = "Subtask 4: Evaluate the final output against predefined criteria to determine its correctness and reliability."
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc4
    )
    agents.append(f"Review agent {results4['review_agent'].id}, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Sub-task 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    reflexion_instruction5 = "Subtask 5: Based on the review feedback, refine the final answer to ensure correctness and reliability."
    reflexion_desc5 = {
        'instruction': reflexion_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['feedback'], results4['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3', 'feedback of subtask 4', 'correctness of subtask 4']
    }
    critic_desc5 = {
        'instruction': "Please review the refined final answer and provide any remaining limitations.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=reflexion_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
