async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    reasoning_outputs = []
    # Control Flow 0: start_sequential
    # Control Flow 1: start_loop
    for i in range(3):
        cot_instruction = f"Subtask {i+1}: Decompose the passage and question into logical reasoning steps to identify the start and end years of the Yamethin rebellion and calculate its duration."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results = await self.cot(
            subtask_id=f"subtask_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results['cot_agent'].id}, decomposing passage and question, thinking: {results['thinking'].content}; answer: {results['answer'].content}")
        sub_tasks.append(f"Subtask {i+1} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        reasoning_outputs.append(results['answer'].content)
    # Control Flow 2: end_loop
    # Stage 1: Aggregate multiple variant outputs and select the most coherent and consistent final result
    aggregate_instruction = "Subtask 4: From the multiple reasoning outputs, aggregate these solutions and select the most coherent and consistent final result for the duration of the Yamethin rebellion."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + reasoning_outputs,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from previous subtasks"]
    }
    results_agg = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, aggregating reasoning outputs, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    # Stage 2: Iterative Quality Enhancement using Reflexion
    cot_reflect_instruction = "Subtask 5: Iteratively evaluate and refine the selected result to enhance clarity, consistency, and completeness of the answer about the Yamethin rebellion duration."
    critic_instruction = "Please review the refined answer and provide feedback on its clarity, consistency, and completeness."
    cot_reflect_desc = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_reflexion = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, refining answer, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results_reflexion['list_feedback']))):
        agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, feedback: {results_reflexion['list_feedback'][i].content}; correction: {results_reflexion['list_correct'][i].content}")
        if i + 1 < len(results_reflexion['list_thinking']) and i + 1 < len(results_reflexion['list_answer']):
            agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, refining further, thinking: {results_reflexion['list_thinking'][i+1].content}; answer: {results_reflexion['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
    logs.append(results_reflexion['subtask_desc'])
    # Stage 3: Validate and Transform Output
    review_instruction = "Subtask 6: Review the refined answer to ensure correctness and completeness regarding the Yamethin rebellion duration."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results_review = await self.review(
        subtask_id="subtask_6",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing refined answer, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Subtask 6 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    formatter_instruction = "Subtask 7: Format the final answer concisely and clearly as the number of years the Yamethin rebellion lasted."
    formatter_desc = {
        'instruction': formatter_instruction,
        'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'format': 'short and concise, without explanation'
    }
    results_format = await self.specific_format(
        subtask_id="subtask_7",
        formatter_desc=formatter_desc
    )
    agents.append(f"SpecificFormat agent {results_format['formatter_agent'].id}, formatting final answer, thinking: {results_format['thinking'].content}; answer: {results_format['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results_format['thinking'].content}; answer - {results_format['answer'].content}")
    logs.append(results_format['subtask_desc'])
    # Control Flow 3: end_sequential
    final_answer = await self.make_final_answer(results_format['thinking'], results_format['answer'], sub_tasks, agents)
    return final_answer, logs