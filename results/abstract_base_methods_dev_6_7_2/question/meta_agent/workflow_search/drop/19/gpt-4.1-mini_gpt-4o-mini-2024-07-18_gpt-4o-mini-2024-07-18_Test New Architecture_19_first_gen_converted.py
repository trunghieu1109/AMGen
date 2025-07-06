async def forward_19(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    
    loop_iterations = 3
    for i in range(loop_iterations):
        
        # Stage 0 - Subtask 1: Decompose passage into logical sequence (CoT | AnswerGenerate)
        cot_instruction1 = "Subtask 1: Decompose the passage information into an ordered logical sequence to identify which population groups have between 10% and 15% of the population, producing an initial answer."
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
        agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing passage, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
        sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
        logs.append(results1['subtask_desc'])
        
        # Stage 1 - Subtask 2: Iteratively evaluate and improve logical reasoning sequence (Reflexion | Revise)
        revise_instruction2 = "Subtask 2: Iteratively evaluate and improve the logical reasoning sequence that identifies population groups with percentages between 10% and 15%, enhancing clarity, consistency, and completeness."
        revise_desc2 = {
            'instruction': revise_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results2 = await self.reflexion(
            subtask_id="subtask_2",
            cot_reflect_desc=revise_desc2,
            critic_desc={
                'instruction': "Please review the logical reasoning sequence and provide its limitations.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining reasoning, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results2['list_feedback']))):
            agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correction: {results2['list_correct'][i].content}")
            if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
                agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining final answer, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
    
    # Control Flow 2: end_loop
    
    # Stage 3 - Subtask 3: Optionally evaluate the constructed answer (Review | Reflexion)
    review_instruction3 = "Subtask 3: Evaluate the constructed answer against correctness and reliability criteria to validate the output."
    review_desc3 = {
        'instruction': review_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2']
    }
    results3 = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc3
    )
    agents.append(f"Review agent {results3['review_agent'].id}, review solution, feedback: {results3['feedback'].content}; correct: {results3['correct'].content}")
    sub_tasks.append(f"Subtask 3 output: feedback - {results3['feedback'].content}; correct - {results3['correct'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 2 - Subtask 4: Aggregate multiple candidate answers and select the most coherent final result (Aggregate)
    aggregate_instruction4 = "Subtask 4: Aggregate multiple candidate answers and select the most coherent and consistent final result identifying the groups with 10%-15% population share."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results1['answer'], results2['answer'], results3['feedback']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from previous subtasks']
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating answers, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs