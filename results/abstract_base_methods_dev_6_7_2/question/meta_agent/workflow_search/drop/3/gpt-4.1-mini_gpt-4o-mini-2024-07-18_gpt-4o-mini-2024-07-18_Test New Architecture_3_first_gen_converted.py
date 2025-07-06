async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Start sequential flow
    # Start loop flow for iterative quality enhancement
    for i in range(3):
        cot_reflect_instruction = f"Subtask 1: Iteratively evaluate and improve the initial interpretation of the passage to enhance clarity, consistency, and completeness regarding the regions where the uprising started, iteration {i+1}."
        critic_instruction = f"Subtask 1: Review the revised interpretation for iteration {i+1} and provide feedback on limitations."
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
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correct: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining final answer, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        sub_tasks.append(f"Subtask 1 iteration {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
    # End loop flow
    # Stage 0: Decompose passage into logical sequence to determine exact number of regions
    cot_instruction2 = "Subtask 2: Decompose the passage information into an ordered logical sequence to determine the exact number of regions where the popular uprising started."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo] + [results_reflexion['thinking'], results_reflexion['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking and answer from iterative refinement"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, decomposing passage, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    # Stage 3 (optional): Evaluate constructed answer for correctness and reliability
    review_instruction = "Subtask 3: Evaluate the constructed answer for correctness and reliability against the passage details."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking and answer from subtask 2"]
    }
    results3 = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results3['review_agent'].id}, feedback: {results3['feedback'].content}; correct: {results3['correct'].content}")
    sub_tasks.append(f"Subtask 3 output: feedback - {results3['feedback'].content}; correct - {results3['correct'].content}")
    logs.append(results3['subtask_desc'])
    # Stage 2: Aggregate outputs and select most coherent final answer
    aggregate_instruction = "Subtask 4: Aggregate the outputs from reasoning and validation steps to select the most coherent and consistent final answer to the question about the number of regions where the popular uprising started."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['feedback'], results3['correct']],
        'temperature': 0.0,
        'context': ["user query", "solutions from subtask 2 and review from subtask 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs