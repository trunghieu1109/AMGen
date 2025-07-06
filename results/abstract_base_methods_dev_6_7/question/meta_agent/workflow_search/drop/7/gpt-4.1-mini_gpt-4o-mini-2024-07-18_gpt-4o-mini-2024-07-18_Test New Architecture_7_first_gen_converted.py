async def forward_7(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    extracted_mentions = []
    # Control Flow 0: start_sequential
    # Control Flow 1: start_loop
    for i in range(3):
        cot_instruction = f"Subtask 1: Decompose the passage to identify and extract mentions of the 49ers quarterback and related actions in an ordered logical sequence, iteration {i+1}."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results = await self.cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results['cot_agent'].id}, iteration {i+1}, thinking: {results['thinking'].content}; answer: {results['answer'].content}")
        sub_tasks.append(f"Subtask 1 iteration {i+1} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        extracted_mentions.append(results['answer'].content)
    # Control Flow 2: end_loop
    # Stage 1: Aggregate extracted mentions
    aggregate_instruction = "Subtask 2: Aggregate the extracted mentions and reasoning steps to select the most coherent and consistent identification of the 49ers quarterback."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + extracted_mentions,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 1"]
    }
    results2 = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results2['aggregate_agent'].id}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    # Stage 2: Iterative quality enhancement with Reflexion
    cot_reflect_instruction = "Subtask 3: Iteratively evaluate and refine the selected answer to enhance clarity, consistency, and completeness regarding the 49ers quarterback identity."
    critic_instruction = "Please review the refined answer and provide feedback on its clarity and correctness."
    cot_reflect_desc = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correct: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining final answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    # Stage 3: Validate and transform output
    review_instruction = "Subtask 4: Review the refined answer for correctness and transform it to conform to the specified output format (direct answer to the question)."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results4['review_agent'].id}, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    specific_format_instruction = "Subtask 5: Format the final answer to be short and concise, without explanation, as the direct answer to the question 'Who is the 49ers quarterback?'."
    specific_format_desc = {
        'instruction': specific_format_instruction,
        'input': [taskInfo, results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 3"],
        'format': 'short and concise, without explaination'
    }
    results5 = await self.specific_format(
        subtask_id="subtask_5",
        formatter_desc=specific_format_desc
    )
    agents.append(f"SpecificFormat agent {results5['formatter_agent'].id}, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs