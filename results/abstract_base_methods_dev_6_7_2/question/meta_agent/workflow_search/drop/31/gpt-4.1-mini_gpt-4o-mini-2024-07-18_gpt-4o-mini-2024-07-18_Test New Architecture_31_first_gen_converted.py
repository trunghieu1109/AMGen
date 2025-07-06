async def forward_31(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    refined_outputs = []
    loop_iterations = 3
    for i in range(loop_iterations):
        reflexion_instruction = f"Subtask 1: Iteratively evaluate and refine the extraction of relevant information about the Siamese forces in 1852 from the passage to improve clarity, consistency, and completeness. Iteration {i+1}."
        reflexion_critic_instruction = "Please review the refined information extraction and provide its limitations."
        reflexion_desc = {
            'instruction': reflexion_instruction,
            'input': [taskInfo],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc = {
            'instruction': reflexion_critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id=f"subtask_{i+1}",
            cot_reflect_desc=reflexion_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correction: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1} refinement, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        refined_outputs.append(results_reflexion['answer'].content)
    cot_instruction2 = "Sub-task 4: Decompose the refined information into an ordered logical reasoning sequence to determine the number of animal forces the Siamese took with them in 1852."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, refined_outputs],
        'temperature': 0.0,
        'context': ["user query", "refined information from iterative reflexion"]
    }
    results_cot = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, reasoning on refined information, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    final_answer = await self.make_final_answer(results_cot['thinking'], results_cot['answer'], sub_tasks, agents)
    return final_answer, logs