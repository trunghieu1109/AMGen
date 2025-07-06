async def forward_20(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose input information into an ordered sequence of logical steps to derive an initial outcome for the number of years between 1990 and 1999 that changed the annual total employment."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.answer_generate(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing input, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Stage 1: Iterative Quality Enhancement
    for i in range(self.max_round):
        cot_reflect_instruction2 = "Subtask 2: Iteratively evaluate and modify the initial reasoning sequence to enhance clarity, consistency, and completeness of the answer regarding the number of years between 1990 and 1999 affecting employment change."
        critic_instruction2 = "Please review the reasoning sequence and provide feedback on its clarity, consistency, and completeness."
        cot_reflect_desc2 = {
            'instruction': cot_reflect_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        critic_desc2 = {
            'instruction': critic_instruction2,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results2 = await self.reflexion(
            subtask_id="subtask_2",
            cot_reflect_desc=cot_reflect_desc2,
            critic_desc=critic_desc2,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining answer, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][0].content}; correction: {results2['list_correct'][0].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results2['list_thinking'][0].content}; revised_solution - {results2['list_answer'][0].content}")
        logs.append(results2['subtask_desc'])
        results1 = {
            'thinking': results2['list_thinking'][0],
            'answer': results2['list_answer'][0]
        }
    
    # Control Flow 2: end_loop
    
    # Control Flow 3: end_sequential
    
    final_answer = await self.make_final_answer(results1['thinking'], results1['answer'], sub_tasks, agents)
    return final_answer, logs