async def forward_22(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Stage 0: Construct Logical Reasoning Sequence
    # Start Control Flow 1: start_loop
    loop_results = []
    for i in range(self.max_sc):
        cot_instruction = "Sub-task 1: Decompose input information into an ordered sequence of logical steps to derive an initial outcome identifying which team scored first, based on the passage provided."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results1 = await self.sc_cot(
            subtask_id="subtask_1",
            cot_sc_desc=cot_agent_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results1['cot_agent'][0].id}, reasoning about which team scored first, thinking: {results1['list_thinking'][0]}; answer: {results1['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 1 output: thinking - {results1['list_thinking'][0]}; answer - {results1['list_answer'][0]}")
        logs.append(results1['subtask_desc'])
        loop_results.append(results1['list_answer'][0])
    # End Control Flow 2: end_loop
    
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction = "Sub-task 2: Aggregate multiple variant outputs from the logical reasoning sequences and select the most coherent and consistent final result regarding which team scored first."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + loop_results,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 1"]
    }
    results2 = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results2['aggregate_agent'].id}, consolidating reasoning outputs, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    # Stage 2: Iterative Quality Enhancement
    cot_reflect_instruction = "Sub-task 3: Iteratively evaluate and modify the aggregated output to enhance clarity, consistency, and completeness of the answer about which team scored first."
    critic_instruction = "Please review the clarity, consistency, and completeness of the answer identifying which team scored first, and provide feedback."
    cot_reflect_desc = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer about which team scored first, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 3: Validate and Transform Output
    review_instruction = "Sub-task 4: Evaluate the final output to ensure correctness and transform it to conform to the specified answer format (which team scored first)."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing final answer, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Sub-task 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    formatter_instruction = "Sub-task 5: Format the final answer concisely indicating which team scored first, without explanation."
    formatter_desc = {
        'instruction': formatter_instruction,
        'input': [taskInfo, results3['answer']],
        'temperature': 0.0,
        'context': ["user query"],
        'format': 'short and concise, without explanation'
    }
    results5 = await self.specific_format(
        subtask_id="subtask_5",
        formatter_desc=formatter_desc
    )
    agents.append(f"SpecificFormat agent {results5['formatter_agent'].id}, formatting final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs