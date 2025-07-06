async def forward_44(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    combined_results = []
    for i in range(self.max_sc):
        cot_instruction1 = "Subtask 1: Decompose the input passage and question into an ordered sequence of logical steps to calculate the combined market worth of North America, Asia, and Europe."
        cot_agent_desc = {
            'instruction': cot_instruction1,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results1 = await self.sc_cot(
            subtask_id="subtask_1",
            cot_sc_desc=cot_agent_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results1['cot_agent'][0].id}, decomposing and calculating combined market worth, thinking: {results1['list_thinking'][0]}; answer: {results1['list_answer'][0]}")
        sub_tasks.append(f"Subtask 1 output: thinking - {results1['list_thinking'][0]}; answer - {results1['list_answer'][0]}")
        logs.append(results1['subtask_desc'])
        combined_results.append(results1['list_answer'][0])
    
    # Control Flow 2: end_loop
    
    # Stage 1: Aggregate multiple reasoning outputs and select the most coherent and consistent calculation
    aggregate_instruction2 = "Subtask 2: Aggregate multiple reasoning outputs from Subtask 1 and select the most coherent and consistent calculation of the combined market worth."
    aggregate_desc = {
        'instruction': aggregate_instruction2,
        'input': [taskInfo] + combined_results,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 1"]
    }
    results2 = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results2['aggregate_agent'].id}, aggregating solutions, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    # Stage 2: Iteratively evaluate and refine the aggregated calculation
    cot_reflect_instruction3 = "Subtask 3: Iteratively evaluate and refine the aggregated calculation to enhance clarity, consistency, and completeness of the combined market worth result."
    critic_instruction3 = "Please review the refined combined market worth calculation and provide its limitations."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining combined market worth, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining final answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 3: Validate the final combined market worth calculation and transform output to specified format
    review_instruction4 = "Subtask 4: Validate the final combined market worth calculation against the question requirements and transform the output to the specified format (complete numeric answer in billions)."
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc4
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing final combined market worth, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    specific_format_instruction = "Subtask 4: Format the validated combined market worth answer as a concise numeric value in billions, without explanation."
    specific_format_desc = {
        'instruction': specific_format_instruction,
        'input': [taskInfo, results4['feedback'], results4['correct']],
        'temperature': 0.0,
        'context': ["user query", "feedback of subtask 4", "correct of subtask 4"],
        'format': 'short and concise, without explanation'
    }
    results5 = await self.specific_format(
        subtask_id="subtask_5",
        formatter_desc=specific_format_desc
    )
    agents.append(f"SpecificFormat agent {results5['formatter_agent'].id}, formatting final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    
    # Control Flow 3: end_sequential
    
    return final_answer, logs
