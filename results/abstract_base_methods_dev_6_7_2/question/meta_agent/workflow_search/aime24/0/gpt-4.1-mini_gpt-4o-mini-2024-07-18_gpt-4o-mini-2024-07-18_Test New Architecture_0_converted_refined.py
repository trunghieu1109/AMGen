async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Define variables s (walking speed in km/h) and t (coffee stop time in minutes); formulate equations 9/s + t/60 = 4 and 9/(s+2) + t/60 = 2.4 based on the problem context, explicitly confirming units and conversions."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, defining variables and equations, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    candidate_solutions = []
    for i in range(self.max_sc):
        cot_sc_instruction2 = (
            "Sub-task 2: Solve the system 9/s + t/60 = 4 and 9/(s+2) + t/60 = 2.4 to find candidate numerical values for s (km/h) and t (minutes). "
            "Explicitly verify each candidate solution by substituting back into original equations and discard any inconsistent or unrealistic solutions (e.g., negative speeds or times)."
        )
        cot_sc_desc = {
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.sc_cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        candidate_solutions.append((results2['thinking'], results2['answer']))
        agents.append(f"CoT-SC agent {results2['cot_agent'][0].id}, iteration {i+1}, candidate solution thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 2 iteration {i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])

    aggregate_instruction3 = (
        "Sub-task 3: Aggregate candidate solutions for s and t from Sub-task 2, eliminate inconsistent or unrealistic pairs by detailed justification, and produce a single viable solution that satisfies all problem constraints."
    )
    aggregate_desc = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + [ans for _, ans in candidate_solutions],
        'temperature': 0.0,
        'context': ["user query", "candidate solutions from subtask 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating candidates, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    answer_generate_instruction4 = (
        "Sub-task 4: Confirm unit conversions and fill any missing details in the solution for s (km/h) and t (minutes). "
        "Explicitly confirm that t is in minutes and s in km/h, and prepare for final calculation."
    )
    answer_generate_desc = {
        'instruction': answer_generate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated solution"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=answer_generate_desc
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, confirming units and filling gaps, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    review_instruction5 = (
        "Sub-task 5: Validate the computed walking speed s and coffee time t against problem conditions: ensure both are positive, realistic, and consistent with the original problem."
    )
    review_desc = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results5['review_agent'].id}, validating solution, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    if results5['answer'].content.lower() in ['yes', 'true', 'correct', 'valid']:
        cot_reflect_instruction6 = (
            "Sub-task 6: Using validated s and t, calculate total time in minutes when walking at speed s + 0.5 km/h, including coffee stop time t. "
            "Explicitly confirm unit consistency and final rounding."
        )
        critic_instruction6 = "Please review the calculation of total time for walking at s + 0.5 km/h including coffee time t, checking for positivity, realism, and consistency."
        cot_reflect_desc6 = {
            'instruction': cot_reflect_instruction6,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        critic_desc6 = {
            'instruction': critic_instruction6,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results6 = await self.reflexion(
            subtask_id="subtask_6",
            cot_reflect_desc=cot_reflect_desc6,
            critic_desc=critic_desc6,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, calculating total time, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results6['list_feedback']))):
            agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correct: {results6['list_correct'][i].content}")
        sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

        formatter_instruction7 = "Sub-task 7: Refine reasoning and presentation of the total time calculation for clarity and coherence, ensuring no ambiguity in units or interpretation."
        formatter_desc7 = {
            'instruction': formatter_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results7 = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=formatter_desc7
        )
        agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, refining clarity, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        formatter_instruction8 = "Sub-task 8: Format the final answer as an integer number of minutes, output integer only without extra text."
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
            'format': 'integer only'
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, formatting final answer, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

    else:
        formatter_instruction9 = "Sub-task 9: Identify and clarify any ambiguities in units or interpretation of s and t, validate correctness, ensuring consistent units and realistic values."
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results9 = await self.cot(
            subtask_id="subtask_9",
            cot_agent_desc=formatter_desc9
        )
        agents.append(f"CoT agent {results9['cot_agent'].id}, clarifying units, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        answer_generate_instruction10 = "Sub-task 10: Generate revised solution after clarifying units and interpretation, ensuring algebraic correctness and realistic values."
        answer_generate_desc10 = {
            'instruction': answer_generate_instruction10,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results10 = await self.cot(
            subtask_id="subtask_10",
            cot_agent_desc=answer_generate_desc10
        )
        agents.append(f"CoT agent {results10['cot_agent'].id}, revising solution, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
        sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

        review_instruction11 = "Sub-task 11: Validate the revised solution for correctness, consistency, positivity, and proper rounding."
        review_desc11 = {
            'instruction': review_instruction11,
            'input': [taskInfo, results10['thinking'], results10['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 10", "answer of subtask 10"]
        }
        results11 = await self.reflexion(
            subtask_id="subtask_11",
            cot_reflect_desc={
                'instruction': review_instruction11,
                'input': [taskInfo, results10['thinking'], results10['answer']],
                'output': ["thinking", "answer"],
                'temperature': 0.0,
                'context': ["user query", "thinking of subtask 10", "answer of subtask 10"]
            },
            critic_desc={
                'instruction': "Please review the validation and provide feedback and corrections.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results11['cot_agent'].id}, validating revised solution, thinking: {results11['list_thinking'][0].content}; answer: {results11['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results11['list_feedback']))):
            agents.append(f"Critic agent {results11['critic_agent'].id}, feedback: {results11['list_feedback'][i].content}; correct: {results11['list_correct'][i].content}")
        sub_tasks.append(f"Sub-task 11 output: thinking - {results11['thinking'].content}; answer - {results11['answer'].content}")
        logs.append(results11['subtask_desc'])

        formatter_instruction12 = "Sub-task 12: Refine revised solution presentation for clarity and coherence, output concise final answer only."
        formatter_desc12 = {
            'instruction': formatter_instruction12,
            'input': [taskInfo, results11['thinking'], results11['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 11", "answer of subtask 11"]
        }
        results12 = await self.cot(
            subtask_id="subtask_12",
            cot_agent_desc=formatter_desc12
        )
        agents.append(f"CoT agent {results12['cot_agent'].id}, refining clarity, thinking: {results12['thinking'].content}; answer: {results12['answer'].content}")
        sub_tasks.append(f"Sub-task 12 output: thinking - {results12['thinking'].content}; answer - {results12['answer'].content}")
        logs.append(results12['subtask_desc'])

        formatter_instruction13 = "Sub-task 13: Format the final answer as an integer number of minutes after revision, output integer only without extra text."
        formatter_desc13 = {
            'instruction': formatter_instruction13,
            'input': [taskInfo, results12['thinking'], results12['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 12", "answer of subtask 12"],
            'format': 'integer only'
        }
        results13 = await self.specific_format(
            subtask_id="subtask_13",
            formatter_desc=formatter_desc13
        )
        agents.append(f"SpecificFormat agent {results13['formatter_agent'].id}, formatting final answer, thinking: {results13['thinking'].content}; answer: {results13['answer'].content}")
        sub_tasks.append(f"Sub-task 13 output: thinking - {results13['thinking'].content}; answer - {results13['answer'].content}")
        logs.append(results13['subtask_desc'])

    if results5['answer'].content.lower() in ['yes', 'true', 'correct', 'valid']:
        final_thinking = results8['thinking']
        final_answer = results8['answer']
    else:
        final_thinking = results13['thinking']
        final_answer = results13['answer']

    final_result = await self.make_final_answer(final_thinking, final_answer, sub_tasks, agents)
    return final_result, logs
