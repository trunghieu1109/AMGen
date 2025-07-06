async def forward_30(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    loop_results = []
    # Control Flow 0: start_sequential
    # Control Flow 1: start_loop
    for i in range(self.max_round):
        # Stage 1: Iterative Quality Enhancement
        cot_reflect_instruction = "Sub-task 1: Iteratively evaluate and refine the understanding of the passage and question to improve clarity, consistency, and completeness of the information needed to answer the question."
        critic_instruction = "Please review the refined understanding and provide feedback on clarity, consistency, and completeness."
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
            subtask_id=f"subtask_1_iter_{i+1}",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][0].content}; correct: {results_reflexion['list_correct'][0].content}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results_reflexion['list_thinking'][0].content}; answer - {results_reflexion['list_answer'][0].content}")
        logs.append(results_reflexion['subtask_desc'])
        # Stage 0: Construct Logical Reasoning Sequence
        cot_instruction = "Sub-task 2: Decompose the refined information into an ordered logical sequence to identify the founding years of the Russian Compound and Mahane Israel and calculate the number of years between them."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo, results_reflexion['list_thinking'][0], results_reflexion['list_answer'][0]],
            'temperature': 0.0,
            'context': ["user query", "refined thinking", "refined answer"]
        }
        results_cot = await self.cot(
            subtask_id=f"subtask_2_iter_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, iteration {i+1}, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Sub-task 2 iteration {i+1} output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
        loop_results.append((results_cot['thinking'], results_cot['answer']))
    # Control Flow 2: end_loop
    # Stage 2: Consolidate and select optimal output
    aggregate_instruction = "Sub-task 3: Aggregate the outputs from the iterative loop and select the most coherent and consistent final result for the number of years between the founding of the Russian Compound and Mahane Israel."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + [item[1] for item in loop_results],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from iterative subtasks"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    # Stage 3: Validate Output (optional)
    if hasattr(self, 'review') and hasattr(self, 'reflexion'):
        review_instruction = "Sub-task 4: Evaluate the aggregated final output against correctness and reliability criteria to validate the answer."
        review_desc = {
            'instruction': review_instruction,
            'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
            'temperature': 0.0,
            'context': ["user query", "aggregated thinking", "aggregated answer"]
        }
        results_review = await self.review(
            subtask_id="subtask_4",
            review_desc=review_desc
        )
        agents.append(f"Review agent {results_review['review_agent'].id}, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
        sub_tasks.append(f"Sub-task 4 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
        logs.append(results_review['subtask_desc'])
        cot_reflect_instruction2 = "Sub-task 5: Based on the review feedback, refine and validate the final answer for the number of years between the founding of the Russian Compound and Mahane Israel."
        critic_instruction2 = "Please review the refined final answer and provide any remaining limitations or confirmations."
        cot_reflect_desc2 = {
            'instruction': cot_reflect_instruction2,
            'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer'], results_review['feedback'], results_review['correct']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "aggregated thinking", "aggregated answer", "review feedback", "review correctness"]
        }
        critic_desc2 = {
            'instruction': critic_instruction2,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion2 = await self.reflexion(
            subtask_id="subtask_5",
            cot_reflect_desc=cot_reflect_desc2,
            critic_desc=critic_desc2,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results_reflexion2['cot_agent'].id}, thinking: {results_reflexion2['list_thinking'][0].content}; answer: {results_reflexion2['list_answer'][0].content}")
        agents.append(f"Critic agent {results_reflexion2['critic_agent'].id}, feedback: {results_reflexion2['list_feedback'][0].content}; correct: {results_reflexion2['list_correct'][0].content}")
        sub_tasks.append(f"Sub-task 5 output: thinking - {results_reflexion2['list_thinking'][0].content}; answer - {results_reflexion2['list_answer'][0].content}")
        logs.append(results_reflexion2['subtask_desc'])
        final_thinking = results_reflexion2['list_thinking'][0]
        final_answer = results_reflexion2['list_answer'][0]
    else:
        final_thinking = results_aggregate['thinking']
        final_answer = results_aggregate['answer']
    final_result = await self.make_final_answer(final_thinking, final_answer, sub_tasks, agents)
    return final_result, logs