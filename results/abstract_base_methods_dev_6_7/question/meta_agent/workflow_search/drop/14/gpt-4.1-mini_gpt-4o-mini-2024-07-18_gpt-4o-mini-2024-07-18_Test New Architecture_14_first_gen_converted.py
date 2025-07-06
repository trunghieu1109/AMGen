async def forward_14(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    for iteration in range(3):
        cot_reflect_instruction = "Subtask 1: Iteratively evaluate and refine the extraction and filtering of racial groups from the passage to ensure clarity, consistency, and completeness in identifying groups with population percentages between 0.5% and 2.5%."
        critic_instruction = "Please review the refined racial group extraction and filtering and provide its limitations."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id="subtask_1", 
            cot_reflect_desc=cot_reflect_desc, 
            critic_desc=critic_desc, 
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, filter valid racial groups, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, providing feedback, thinking: {results_reflexion['list_feedback'][i].content}; answer: {results_reflexion['list_correct'][i].content}")
            if i + 1 < len(results_reflexion['list_thinking']) and i + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, refining final answer, thinking: {results_reflexion['list_thinking'][i + 1].content}; answer: {results_reflexion['list_answer'][i + 1].content}")
        sub_tasks.append(f"Sub-task 1 output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        
        revise_instruction = "Subtask 2: Revise previous refined racial group extraction and filtering to improve clarity, consistency, and completeness."
        revise_desc = {
            'instruction': revise_instruction,
            'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results_revise = await self.revise(
            subtask_id="subtask_2", 
            revise_desc=revise_desc
        )
        agents.append(f"Revise agent {results_revise['revise_agent'].id}, revise solution from subtask 1, feedback: [feedback], thinking: {results_revise['thinking'].content}; revised_solution: {results_revise['revised_solution'].content}")
        sub_tasks.append(f"Sub-task 2 output: thinking - {results_revise['thinking'].content}; revised_solution - {results_revise['revised_solution'].content}")
        logs.append(results_revise['subtask_desc'])
        
    # Control Flow 2: end_loop
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction_final = "Subtask 3: Decompose the refined racial group data into an ordered logical reasoning sequence to identify which groups individually make up between 0.5% and 2.5% of the population and generate the final answer."
    cot_agent_desc_final = {
        'instruction': cot_instruction_final,
        'input': [taskInfo, results_revise['thinking'], results_revise['revised_solution']],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_final = await self.cot(
        subtask_id="subtask_3", 
        cot_agent_desc=cot_agent_desc_final
    )
    agents.append(f"CoT agent {results_final['cot_agent'].id}, decomposing refined data, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    
    # Control Flow 3: end_sequential
    
    return final_answer, logs