async def forward_19(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    loop_iterations = 3
    for i in range(loop_iterations):
        cot_reflect_instruction = f"Subtask 1: Iteratively evaluate and improve the initial reasoning about which population groups have between 10% and 15% of the population to enhance clarity, consistency, and completeness. Iteration {i+1}."
        critic_instruction = "Please review the reasoning and provide feedback on clarity, consistency, and completeness."
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
            subtask_id=f"subtask_{i+1}",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correct: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining answer, thinking: {results_reflexion['list_thinking'][k+1].content}; answer: {results_reflexion['list_answer'][k+1].content}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
    
    # Control Flow 2: end_loop
    
    # Stage 0: Decompose passage information into logical sequence using CoT and AnswerGenerate
    cot_instruction = "Sub-task 4: Decompose the passage information into an ordered logical sequence to identify population groups with percentages between 10% and 15%."
    cot_agent_desc = {
        'instruction': cot_instruction,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_cot = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, decomposing passage, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    
    # Stage 3: Validate output using Review and Reflexion
    review_instruction = "Sub-task 5: Evaluate the constructed answer against correctness and reliability criteria to ensure the identified groups are accurate."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results_review = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing answer, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    
    cot_reflect_instruction2 = "Sub-task 6: Based on the output from Sub-task 5, refine and validate the answer to ensure correctness and reliability."
    critic_instruction2 = "Please review the refined answer and provide feedback on any remaining issues."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer'], results_review['feedback'], results_review['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "feedback of subtask 5", "correct of subtask 5"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_reflexion2 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_reflexion2['cot_agent'].id}, refining validated answer, thinking: {results_reflexion2['list_thinking'][0].content}; answer: {results_reflexion2['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results_reflexion2['list_feedback']))):
        agents.append(f"Critic agent {results_reflexion2['critic_agent'].id}, feedback: {results_reflexion2['list_feedback'][k].content}; correct: {results_reflexion2['list_correct'][k].content}")
        if k + 1 < len(results_reflexion2['list_thinking']) and k + 1 < len(results_reflexion2['list_answer']):
            agents.append(f"Reflexion CoT agent {results_reflexion2['cot_agent'].id}, refining final answer, thinking: {results_reflexion2['list_thinking'][k+1].content}; answer: {results_reflexion2['list_answer'][k+1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results_reflexion2['thinking'].content}; answer - {results_reflexion2['answer'].content}")
    logs.append(results_reflexion2['subtask_desc'])
    
    # Stage 2: Aggregate multiple answer variants and select the most coherent and consistent final result
    aggregate_instruction = "Sub-task 7: Aggregate multiple answer variants and select the most coherent and consistent final result identifying the groups with 10%-15% population share."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results_reflexion['answer'], results_cot['answer'], results_reflexion2['answer']],
        'temperature': 0.0,
        'context': ["user query", "answers from previous subtasks"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_7",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating answers, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs
