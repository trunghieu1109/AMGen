async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Start sequential flow
    
    # Start loop flow for iterative refinement of ranking info extraction
    loop_iterations = 3
    refined_thinking = None
    refined_answer = None
    for i in range(loop_iterations):
        cot_reflect_instruction = f"Subtask 1: Iteratively evaluate and refine the extraction of ranking information for 'My Love is Your Love' in Belgium and Germany from the passage to ensure clarity, consistency, and completeness. Iteration {i+1}."
        critic_instruction = "Please review the refined extraction and provide its limitations."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo] + ([refined_thinking, refined_answer] if refined_thinking and refined_answer else []),
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"] + ([refined_thinking.content, refined_answer.content] if refined_thinking and refined_answer else [])
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id=f"subtask_1_iteration_{i+1}",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining extraction, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correction: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining final answer, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        refined_thinking = results_reflexion['list_thinking'][0]
        refined_answer = results_reflexion['list_answer'][0]
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {refined_thinking.content}; answer - {refined_answer.content}")
        logs.append(results_reflexion['subtask_desc'])
    
    # End loop flow
    
    # Subtask 2: Decompose extracted ranking info into logical reasoning sequence and generate final answer
    cot_instruction2 = "Subtask 2: Decompose the extracted ranking information into a logical reasoning sequence to determine whether 'My Love is Your Love' ranked higher in Belgium or Germany, and generate the final answer."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, refined_thinking, refined_answer],
        'temperature': 0.0,
        'context': ["user query", refined_thinking.content, refined_answer.content]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, decomposing ranking info, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    final_answer = await self.make_final_answer(results2['thinking'], results2['answer'], sub_tasks, agents)
    
    # End sequential flow
    
    return final_answer, logs