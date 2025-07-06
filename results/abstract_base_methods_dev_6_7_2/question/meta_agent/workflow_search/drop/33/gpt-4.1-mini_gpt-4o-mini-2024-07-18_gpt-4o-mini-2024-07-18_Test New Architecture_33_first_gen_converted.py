async def forward_33(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the passage into an ordered sequence of scoring events and identify which team scored in the second quarter."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing passage, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    # Control Flow 1: start_loop
    loop_iterations = 3
    refined_thinking = None
    refined_answer = None
    for i in range(loop_iterations):
        # Stage 3: Validate and Transform Output
        review_instruction3 = "Subtask 3: Validate the answer against the passage and format it to conform to the specified output requirements."
        review_desc3 = {
            'instruction': review_instruction3,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results3 = await self.review(
            subtask_id="subtask_3",
            review_desc=review_desc3
        )
        agents.append(f"Review agent {results3['review_agent'].id}, review solution from subtask 1, feedback: {results3['feedback'].content}; correct: {results3['correct'].content}")
        sub_tasks.append(f"Subtask 3 output: feedback - {results3['feedback'].content}; correct - {results3['correct'].content}")
        logs.append(results3['subtask_desc'])

        # Stage 2: Iterative Quality Enhancement
        cot_reflect_instruction2 = "Subtask 2: Iteratively evaluate and revise the initial answer to enhance clarity, consistency, and completeness regarding which team scored in the second quarter."
        revise_desc2 = {
            'instruction': cot_reflect_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer'], results3['feedback'], results3['correct']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'feedback of subtask 3', 'correct of subtask 3']
        }
        results2 = await self.reflexion(
            subtask_id="subtask_2",
            cot_reflect_desc=revise_desc2,
            critic_desc={
                'instruction': "Please review the revised answer and provide its limitations.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining answer, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results2['list_feedback']))):
            agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][k].content}; correct: {results2['list_correct'][k].content}")
            if k + 1 < len(results2['list_thinking']) and k + 1 < len(results2['list_answer']):
                agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining final answer, thinking: {results2['list_thinking'][k + 1].content}; answer: {results2['list_answer'][k + 1].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; revised_solution - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])

        refined_thinking = results2['thinking']
        refined_answer = results2['answer']

    # Control Flow 2: end_loop

    # Stage 1: Consolidate and select optimal output
    aggregate_instruction1 = "Subtask 4: Aggregate multiple variant outputs and select the most coherent and consistent final result about the scoring team in the second quarter."
    aggregate_desc1 = {
        'instruction': aggregate_instruction1,
        'input': [taskInfo, refined_thinking, refined_answer],
        'temperature': 0.0,
        'context': ['user query', 'refined answers from subtask 2']
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc1
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating refined answers, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    # Stage 3: Validate and Transform Output (final formatting)
    formatter_instruction5 = "Subtask 5: Format the final answer to conform to the specified output requirements, providing only the team that scored in the second quarter concisely."
    formatter_desc5 = {
        'instruction': formatter_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ['user query', 'aggregated answer from subtask 4'],
        'format': 'short and concise, without explanation'
    }
    results5 = await self.specific_format(
        subtask_id="subtask_5",
        formatter_desc=formatter_desc5
    )
    agents.append(f"SpecificFormat agent {results5['formatter_agent'].id}, formatting final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs