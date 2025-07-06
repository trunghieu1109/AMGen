async def forward_29(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the input passage and question into an ordered sequence of logical steps to calculate the number of people of German ancestry and those from US or Danish ancestry, then find how many more Germans there were."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing input passage and question, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    # Stage 1: Iterative Quality Enhancement
    iteration_results = []
    for i in range(self.max_round):
        cot_reflect_instruction2 = "Subtask 2: Iteratively evaluate and refine the initial calculation and reasoning to improve clarity, consistency, and completeness of the answer."
        critic_instruction2 = "Please review the refined answer and provide feedback on its clarity, consistency, and completeness."
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
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][0].content}; correct: {results2['list_correct'][0].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results2['list_thinking'][0].content}; revised_solution - {results2['list_answer'][0].content}")
        logs.append(results2['subtask_desc'])
        iteration_results.append((results2['list_thinking'][0], results2['list_answer'][0]))
    
    # Control Flow 2: end_loop
    
    # Stage 2: Consolidate and select optimal output
    aggregate_instruction3 = "Subtask 3: Aggregate the refined outputs from the iterative loop and select the most coherent and consistent final answer to the question."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + [ans for _, ans in iteration_results],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from iterative refinement"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating refined answers, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Control Flow 3: end_sequential
    
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs