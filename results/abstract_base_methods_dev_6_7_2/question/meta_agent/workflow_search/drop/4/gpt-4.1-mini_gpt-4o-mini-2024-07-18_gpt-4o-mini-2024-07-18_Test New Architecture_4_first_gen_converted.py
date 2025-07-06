async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Control Flow 1: start_loop
    loop_iterations = 2
    for i in range(loop_iterations):
        # Stage 1: Iterative Quality Enhancement
        # Subtask 1: Iteratively evaluate and modify an existing logical reasoning sequence using Reflexion and Revise
        if i == 0:
            # Stage 0: Construct Logical Reasoning Sequence
            # Subtask 1: Decompose input information into ordered logical steps using CoT and AnswerGenerate
            cot_instruction1 = "Subtask 1: Decompose input information into an ordered sequence of logical steps to determine which team lost the game in November 1989."
            cot_agent_desc = {
                'instruction': cot_instruction1,
                'input': [taskInfo],
                'temperature': 0.0,
                'context': ["user query"]
            }
            results1 = await self.answer_generate(
                subtask_id="subtask_1",
                cot_agent_desc=cot_agent_desc
            )
            agents.append(f"CoT agent {results1['cot_agent'].id}, decomposing input, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
            sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
            logs.append(results1['subtask_desc'])
            reasoning_thinking = results1['thinking']
            reasoning_answer = results1['answer']
        else:
            # Subtask 2: Reflexion and Revise to improve reasoning
            cot_reflect_instruction = "Subtask 2: Iteratively evaluate and modify the logical reasoning sequence to enhance clarity, consistency, and completeness."
            critic_instruction = "Please review the reasoning and provide feedback on its limitations."
            cot_reflect_desc = {
                'instruction': cot_reflect_instruction,
                'input': [taskInfo, reasoning_thinking, reasoning_answer],
                'output': ["thinking", "answer"],
                'temperature': 0.0,
                'context': ["user query", "previous thinking", "previous answer"]
            }
            critic_desc = {
                'instruction': critic_instruction,
                'output': ["feedback", "correct"],
                'temperature': 0.0
            }
            results2 = await self.reflexion(
                subtask_id="subtask_2",
                cot_reflect_desc=cot_reflect_desc,
                critic_desc=critic_desc,
                n_repeat=self.max_round
            )
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining reasoning, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
            for i in range(min(self.max_round, len(results2['list_feedback']))):
                agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correct: {results2['list_correct'][i].content}")
                if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
                    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining final answer, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
            sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
            logs.append(results2['subtask_desc'])
            reasoning_thinking = results2['thinking']
            reasoning_answer = results2['answer']
    # Control Flow 2: end_loop
    
    # Stage 2: Consolidate and select optimal output
    # Subtask 3: Aggregate multiple variant outputs and select the most coherent and consistent final result
    aggregate_instruction = "Subtask 3: Aggregate multiple reasoning variants and select the most coherent and consistent final answer to which team lost the game in November 1989."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, reasoning_thinking, reasoning_answer],
        'temperature': 0.0,
        'context': ["user query", "reasoning variants"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating variants, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 3: Validate Output
    # Subtask 4: Review and reflexion to evaluate correctness and reliability
    review_instruction = "Subtask 4: Review the aggregated answer for correctness and reliability."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing aggregated answer, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_reflect_instruction4 = "Subtask 4b: Reflexion to finalize the answer based on review feedback."
    critic_instruction4 = "Please provide final feedback and corrections if needed."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['feedback'], results4['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer", "review feedback"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, finalizing answer, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, final feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    # Control Flow 3: end_sequential
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs