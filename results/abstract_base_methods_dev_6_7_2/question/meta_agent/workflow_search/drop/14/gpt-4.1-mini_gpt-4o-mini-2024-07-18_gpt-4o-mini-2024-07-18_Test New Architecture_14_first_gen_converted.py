async def forward_14(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Control Flow 1: start_loop
    # Stage 1: Iterative Quality Enhancement
    loop_iterations = 3
    for i in range(loop_iterations):
        cot_reflect_instruction = "Subtask 1: Iteratively evaluate and refine the extraction of racial groups and their population percentages to ensure clarity, consistency, and completeness in identifying groups within the 0.5% to 2.5% range."
        critic_instruction = "Please review the refined extraction and provide feedback on its clarity, consistency, and completeness."
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
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction = "Sub-task 2: Decompose the census data into a logical sequence of steps to identify racial groups whose population percentages fall between 0.5% and 2.5%, and generate the final list of these groups."
    cot_agent_desc = {
        'instruction': cot_instruction,
        'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results_cot = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, decomposing census data, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    
    # Control Flow 2: end_loop
    # Control Flow 3: end_sequential
    final_answer = await self.make_final_answer(results_cot['thinking'], results_cot['answer'], sub_tasks, agents)
    return final_answer, logs