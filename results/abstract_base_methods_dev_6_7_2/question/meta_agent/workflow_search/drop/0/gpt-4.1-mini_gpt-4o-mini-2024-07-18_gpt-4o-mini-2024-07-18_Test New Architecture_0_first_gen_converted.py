async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the passage information into an ordered logical sequence to determine the start and end years of the Yamethin rebellion and calculate its duration."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, decomposing passage and calculating duration, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    loop_iterations = 3
    refined_thinking = results1['thinking']
    refined_answer = results1['answer']
    for i in range(loop_iterations):
        revise_instruction = f"Subtask 2: Iteratively evaluate and refine the initial reasoning sequence to enhance clarity, consistency, and completeness of the rebellion duration calculation, iteration {i+1}."
        revise_desc = {
            'instruction': revise_instruction,
            'input': [taskInfo, refined_thinking, refined_answer],
            'temperature': 0.0,
            'context': ["user query", "previous thinking", "previous answer"]
        }
        results2 = await self.reflexion(
            subtask_id=f"subtask_{2+i}",
            cot_reflect_desc=revise_desc,
            critic_desc={
                'instruction': f"Please review the refined reasoning and provide feedback for iteration {i+1}.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining reasoning iteration {i+1}, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results2['list_feedback']))):
            agents.append(f"Critic agent {results2['critic_agent'].id}, feedback iteration {i+1}, thinking: {results2['list_feedback'][k].content}; answer: {results2['list_correct'][k].content}")
            if k + 1 < len(results2['list_thinking']) and k + 1 < len(results2['list_answer']):
                agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining final answer iteration {i+1}, thinking: {results2['list_thinking'][k + 1].content}; answer: {results2['list_answer'][k + 1].content}")
        refined_thinking = results2['list_thinking'][0]
        refined_answer = results2['list_answer'][0]
        sub_tasks.append(f"Subtask {2+i} output: thinking - {refined_thinking.content}; answer - {refined_answer.content}")
        logs.append(results2['subtask_desc'])
    
    # Control Flow 2: end_loop
    
    # Stage 3: Validate and Transform Output
    review_instruction = "Subtask 5: Validate the refined output against correctness criteria and transform it to conform to the required answer format (number of years)."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, refined_thinking, refined_answer],
        'temperature': 0.0,
        'context': ["user query", "refined thinking", "refined answer"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing refined output, feedback: {results5['feedback'].content}; correct: {results5['correct'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['feedback'].content}; correct - {results5['correct'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_reflect_instruction = "Subtask 6: Based on the review output, iteratively evaluate and modify the validated output to ensure the final answer is coherent, consistent, and complete."
    critic_instruction = "Please review the modifications and provide feedback."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo, refined_thinking, refined_answer, results5['feedback'], results5['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "refined thinking", "refined answer", "review feedback", "review correctness"]
    }
    critic_desc6 = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc6,
        critic_desc=critic_desc6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback on final refinement, thinking: {results6['list_feedback'][i].content}; answer: {results6['list_correct'][i].content}")
        if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, further refining final answer, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    # Stage 4: Iterative Quality Enhancement (optional)
    cot_reflect_instruction7 = "Subtask 7: Iteratively evaluate and modify the initial logical reasoning sequence to enhance clarity, consistency, and completeness, based on the original passage."
    revise_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "initial thinking", "initial answer"]
    }
    results7 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=revise_desc7,
        critic_desc={
            'instruction': "Please review the iterative refinement and provide feedback.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, iterative refinement of initial reasoning, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback on iterative refinement, thinking: {results7['list_feedback'][i].content}; answer: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, further refining, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction = "Subtask 8: Aggregate the final refined and validated outputs to produce the definitive answer to the question: How many years did the Yamethin rebellion last?"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results6['thinking'], results6['answer'], results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "refined final answers"]
    }
    results8 = await self.aggregate(
        subtask_id="subtask_8",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results8['aggregate_agent'].id}, aggregating final answers, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])
    
    final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
    return final_answer, logs
