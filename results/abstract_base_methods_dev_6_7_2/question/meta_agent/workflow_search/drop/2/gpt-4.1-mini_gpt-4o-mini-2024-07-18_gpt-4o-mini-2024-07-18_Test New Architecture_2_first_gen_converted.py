async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose input information into an ordered sequence of logical steps to derive an initial outcome comparing the chart rankings of 'My Love is Your Love' in Belgium and Germany."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing chart rankings, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    refined_thinkings = []
    refined_answers = []
    loop_iterations = self.max_round
    for i in range(loop_iterations):
        revise_instruction2 = "Subtask 2: Iteratively evaluate and modify the initial reasoning and answer to enhance clarity, consistency, and completeness regarding the ranking comparison between Belgium and Germany."
        revise_desc2 = {
            'instruction': revise_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']] + refined_thinkings + refined_answers,
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'] + refined_thinkings + refined_answers
        }
        results2 = await self.reflexion(
            subtask_id="subtask_2",
            cot_reflect_desc=revise_desc2,
            critic_desc={
                'instruction': "Please review the revised reasoning and provide feedback.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=1
        )
        agents.append(f"Reflexion agent {results2['cot_agent'].id}, refining reasoning, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        for idx in range(len(results2['list_feedback'])):
            agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][idx].content}; correct: {results2['list_correct'][idx].content}")
        refined_thinkings.append(results2['list_thinking'][0].content)
        refined_answers.append(results2['list_answer'][0].content)
        sub_tasks.append(f"Subtask 2 iteration {i+1} output: thinking - {results2['list_thinking'][0].content}; answer - {results2['list_answer'][0].content}")
        logs.append(results2['subtask_desc'])
    
    # Control Flow 2: end_loop
    
    # Stage 1: Consolidate and select optimal output
    aggregate_instruction3 = "Subtask 3: Aggregate multiple refined outputs and select the most coherent and consistent final result answering whether the song ranked higher in Belgium or Germany."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + refined_thinkings + refined_answers,
        'temperature': 0.0,
        'context': ["user query", "refined solutions from subtask 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating refined outputs, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 2: Validate Output (optional)
    review_instruction4 = "Subtask 4: Evaluate the final output against correctness and reliability criteria to optionally validate the answer."
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc4
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing final output, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    # Control Flow 3: end_sequential
    
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs
