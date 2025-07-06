async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the passage to identify and list all distinct regions where the popular uprising started, providing detailed reasoning and initial outcome."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, decomposing passage to identify regions, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    # Stage 3: Validate and Transform Output
    review_instruction3 = "Subtask 3: Validate the identified regions count against the passage and transform the output to the required answer format."
    review_desc3 = {
        'instruction': review_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=review_desc3,
        critic_desc={
            'instruction': "Please review the validation and transformation of identified regions and provide feedback.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, validating and transforming regions, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 2: Iterative Quality Enhancement
    revise_instruction2 = "Subtask 2: Iteratively evaluate and refine the validated output to enhance clarity, consistency, and completeness."
    revise_desc2 = {
        'instruction': revise_instruction2,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=revise_desc2,
        critic_desc={
            'instruction': "Please review the refined output and provide feedback for further improvement.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining output, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correction: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining further, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    # Control Flow 2: end_loop
    
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction1 = "Subtask 4: Aggregate the refined outputs and select the most coherent and consistent final answer for the number of regions where the popular uprising started."
    aggregate_desc1 = {
        'instruction': aggregate_instruction1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'refined outputs from subtask 2']
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc1
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating refined outputs, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    # Control Flow 3: end_sequential
    
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
