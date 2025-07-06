async def forward_41(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    refined_thinking_list = []
    refined_answer_list = []
    loop_iterations = 3
    for i in range(loop_iterations):
        cot_reflect_instruction = f"Subtask {i+1}: Iteratively evaluate and refine the extraction and interpretation of ancestral group data from the passage to ensure clarity, consistency, and completeness. Iteration {i+1}."
        critic_instruction = f"Subtask {i+1}: Please review the refined ancestral group data extraction and provide limitations or improvements. Iteration {i+1}."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo] + refined_thinking_list + refined_answer_list,
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"] + refined_thinking_list + refined_answer_list
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id=f"subtask_{i+1}",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correction: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        refined_thinking_list.append(results_reflexion['thinking'])
        refined_answer_list.append(results_reflexion['answer'])
    cot_instruction_final = "Sub-task final: Decompose the refined ancestral data into a logical reasoning sequence to determine which group, Swedish or United States, is smaller and generate the final answer."
    cot_agent_desc_final = {
        'instruction': cot_instruction_final,
        'input': [taskInfo] + refined_thinking_list + refined_answer_list,
        'temperature': 0.0,
        'context': ["user query"] + refined_thinking_list + refined_answer_list
    }
    results_final = await self.cot(
        subtask_id="subtask_final",
        cot_agent_desc=cot_agent_desc_final
    )
    agents.append(f"CoT agent {results_final['cot_agent'].id}, final reasoning, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Sub-task final output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs