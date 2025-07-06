async def forward_36(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    for i in range(3):
        cot_reflect_instruction1 = "Subtask 1: Iteratively evaluate and refine the understanding of the passage and question to improve clarity, consistency, and completeness of the interpretation."
        critic_instruction1 = "Please review the refined understanding and provide its limitations."
        cot_reflect_desc1 = {
            'instruction': cot_reflect_instruction1,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc1 = {
            'instruction': critic_instruction1,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results1 = await self.reflexion(
            subtask_id="subtask_1", 
            cot_reflect_desc=cot_reflect_desc1, 
            critic_desc=critic_desc1, 
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining understanding, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][0].content}; correct: {results1['list_correct'][0].content}")
        sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
        logs.append(results1['subtask_desc'])
    
    # Control Flow 2: end_loop
    
    # Stage 0: Decompose input information into logical steps to derive the number of sports that will be an event for the first time
    cot_instruction2 = "Sub-task 2: Decompose input information into an ordered sequence of logical steps to derive the number of sports that will be an event for the first time."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "refined thinking from subtask 1", "refined answer from subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, decomposing input, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    # Control Flow 3: end_sequential
    
    final_answer = await self.make_final_answer(results2['thinking'], results2['answer'], sub_tasks, agents)
    return final_answer, logs