async def forward_35(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Start sequential flow
    
    # Start loop flow for iterative refinement
    loop_iterations = 3
    refined_thinking = None
    refined_answer = None
    for i in range(loop_iterations):
        cot_reflect_instruction = f"Sub-task 1: Iteratively evaluate and refine the extraction and interpretation of demographic data from the passage to ensure clarity, consistency, and completeness for subsequent calculations. Iteration {i+1}."
        critic_instruction = f"Please review the refined demographic data extraction and provide feedback for iteration {i+1}."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo] + ([refined_thinking, refined_answer] if refined_thinking and refined_answer else []),
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"] + (["previous thinking", "previous answer"] if refined_thinking and refined_answer else [])
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
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        refined_thinking = results_reflexion['thinking']
        refined_answer = results_reflexion['answer']
    
    # After refinement, decompose refined data into logical sequence to calculate percentage
    cot_instruction2 = "Sub-task 2: Decompose the refined demographic data into an ordered logical sequence to calculate the percentage of the population that is not 65 years of age or older."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, refined_thinking, refined_answer],
        'temperature': 0.0,
        'context': ["user query", "refined thinking", "refined answer"]
    }
    results_cot = await self.answer_generate(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, decomposing refined data, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    
    # Validate the calculated percentage
    review_instruction = "Sub-task 3: Evaluate the calculated percentage against correctness criteria to ensure the answer is reliable and consistent with the passage data."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results_review = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing calculation, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Sub-task 3 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    
    # Aggregate validated outputs and select the best final result
    aggregate_instruction = "Sub-task 4: Aggregate the validated outputs and select the most coherent and consistent final result for the percentage of the population not 65 years or older."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer'], results_review['feedback'], results_review['correct']],
        'temperature': 0.0,
        'context': ["user query", "validated outputs"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating validated outputs, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs
