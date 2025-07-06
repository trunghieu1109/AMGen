async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    results_cot_list = []
    # Control Flow 0: start sequential
    # Stage 0: Construct Logical Reasoning Sequence
    # Control Flow 1: start loop
    for i in range(self.max_sc):
        cot_instruction = "Sub-task 1: Decompose the passage to identify the start and end years of the Yamethin rebellion and calculate its duration in years."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results = await self.cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results['cot_agent'].id}, iteration {i+1}, reasoning: {results['thinking'].content}; answer: {results['answer'].content}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        results_cot_list.append(results)
    # Control Flow 2: end loop
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction = "Sub-task 2: Aggregate multiple reasoning outputs from Sub-task 1 to select the most coherent and consistent calculation of the Yamethin rebellion duration."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + [r['answer'] for r in results_cot_list],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 1"]
    }
    results2 = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results2['aggregate_agent'].id}, reasoning: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    # Stage 2: Iterative Quality Enhancement
    cot_reflect_instruction = "Sub-task 3: Iteratively evaluate and refine the selected output to enhance clarity, consistency, and completeness of the answer about the Yamethin rebellion duration."
    critic_instruction = "Please review the refined answer for clarity, consistency, and completeness, and provide feedback."
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refined answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer iteration {i+2}, answer: {results3['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    # Stage 3: Validate and Transform Output
    review_instruction = "Sub-task 4: Review the refined answer for correctness and completeness regarding the Yamethin rebellion duration."
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
    sub_tasks.append(f"Sub-task 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    formatter_instruction = "Sub-task 4: Format the final answer concisely to state the duration of the Yamethin rebellion in years, without explanation."
    formatter_desc = {
        'instruction': formatter_instruction,
        'input': [taskInfo, results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 3"],
        'format': 'short and concise, without explanation'
    }
    results5 = await self.specific_format(
        subtask_id="subtask_5",
        formatter_desc=formatter_desc
    )
    agents.append(f"SpecificFormat agent {results5['formatter_agent'].id}, formatted answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs