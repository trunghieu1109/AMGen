async def forward_25(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Identify all elements and configurations of the convex equilateral hexagon ABCDEF with opposite sides parallel, and the triangle formed by extensions of AB, CD, EF with sides 200, 240, 300."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, identifying hexagon elements and triangle sides, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Analyze and verify formal relationships among variables and parameters of the hexagon and triangle, determine parameter values that satisfy constraints, especially the hexagon side length."
    cot_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, analyzing formal relationships, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Subtask 3: Determine the side length of the hexagon by evaluating and quantifying the conditions satisfied by the elements and parameters from previous subtasks."
    critic_instruction3 = "Please review the determination of the hexagon side length and provide any limitations or corrections needed."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, filtering valid hexagon side length scenarios, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, providing feedback, thinking: {results3['list_feedback'][i].content}; answer: {results3['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    subtask4_results = []
    for i in range(3):
        cot_agent_instruction4 = f"Subtask 4.{i+1}: Generate initial candidate hexagon side length solutions by logical reasoning iteration {i+1}."
        cot_agent_desc4 = {
            'instruction': cot_agent_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.3,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        result4 = await self.answer_generate(
            subtask_id=f"subtask_4_{i+1}",
            cot_agent_desc=cot_agent_desc4
        )
        agents.append(f"AnswerGenerate agent {result4['cot_agent'].id}, generating candidate solution iteration {i+1}, thinking: {result4['thinking'].content}; answer: {result4['answer'].content}")
        sub_tasks.append(f"Sub-task 4.{i+1} output: thinking - {result4['thinking'].content}; answer - {result4['answer'].content}")
        logs.append(result4['subtask_desc'])
        subtask4_results.append(result4)

    aggregate_instruction5 = "Subtask 5: Aggregate candidate hexagon side length solutions, evaluate consistency, clarify units, and validate outputs against criteria."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + [r['answer'] for r in subtask4_results],
        'temperature': 0.0,
        'context': ["user query", "candidate solutions from subtask 4"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, aggregating candidate solutions, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_agent_instruction6 = "Subtask 6: Identify and fill any gaps or inconsistencies in the aggregated hexagon side length solution."
    cot_agent_desc6 = {
        'instruction': cot_agent_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.answer_generate(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"AnswerGenerate agent {results6['cot_agent'].id}, filling gaps in solution, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    condition = 'valid_solution' in results6['answer'].content.lower()

    if condition:
        # True branch
        cot_agent_instruction7 = "Subtask 7: Identify discrete units of information from the validated solution and ensure correctness."
        cot_agent_desc7 = {
            'instruction': cot_agent_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"],
            'format': 'short and concise, without explanation'
        }
        results7 = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=cot_agent_desc7
        )
        agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, identifying and validating units, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        review_instruction8 = "Subtask 8: Evaluate and validate the solution against defined criteria for accuracy and completeness."
        review_desc8 = {
            'instruction': review_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
        }
        results8 = await self.review(
            subtask_id="subtask_8",
            review_desc=review_desc8
        )
        agents.append(f"Review agent {results8['review_agent'].id}, validating solution, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        cot_agent_instruction9 = "Subtask 9: Refine and restructure the solution to enhance clarity and coherence."
        cot_agent_desc9 = {
            'instruction': cot_agent_instruction9,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=cot_agent_desc9
        )
        agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, refining solution clarity, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        cot_agent_instruction10 = "Subtask 10: Format the finalized hexagon side length solution to meet output requirements."
        cot_agent_desc10 = {
            'instruction': cot_agent_instruction10,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results10 = await self.specific_format(
            subtask_id="subtask_10",
            formatter_desc=cot_agent_desc10
        )
        agents.append(f"SpecificFormat agent {results10['formatter_agent'].id}, formatting final solution, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
        sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

    else:
        cot_agent_instruction7f = "Subtask 7: Identify discrete units of information from the unvalidated solution and attempt to resolve ambiguities."
        cot_agent_desc7f = {
            'instruction': cot_agent_instruction7f,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"],
            'format': 'short and concise, without explanation'
        }
        results7f = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=cot_agent_desc7f
        )
        agents.append(f"SpecificFormat agent {results7f['formatter_agent'].id}, identifying and clarifying units in false branch, thinking: {results7f['thinking'].content}; answer: {results7f['answer'].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results7f['thinking'].content}; answer - {results7f['answer'].content}")
        logs.append(results7f['subtask_desc'])

        review_instruction8f = "Subtask 8: Evaluate and validate the unvalidated solution, refining as necessary."
        review_desc8f = {
            'instruction': review_instruction8f,
            'input': [taskInfo, results7f['thinking'], results7f['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
        }
        results8f = await self.review(
            subtask_id="subtask_8",
            review_desc=review_desc8f
        )
        agents.append(f"Review agent {results8f['review_agent'].id}, validating false branch solution, feedback: {results8f['thinking'].content}; correct: {results8f['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: feedback - {results8f['thinking'].content}; correct - {results8f['answer'].content}")
        logs.append(results8f['subtask_desc'])

        cot_agent_instruction9f = "Subtask 9: Refine and restructure the false branch solution for clarity and coherence."
        cot_agent_desc9f = {
            'instruction': cot_agent_instruction9f,
            'input': [taskInfo, results8f['thinking'], results8f['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
        }
        results9f = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=cot_agent_desc9f
        )
        agents.append(f"SpecificFormat agent {results9f['formatter_agent'].id}, refining false branch solution, thinking: {results9f['thinking'].content}; answer: {results9f['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9f['thinking'].content}; answer - {results9f['answer'].content}")
        logs.append(results9f['subtask_desc'])

        cot_agent_instruction10f = "Subtask 10: Format the finalized false branch solution to meet output requirements."
        cot_agent_desc10f = {
            'instruction': cot_agent_instruction10f,
            'input': [taskInfo, results9f['thinking'], results9f['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results10f = await self.specific_format(
            subtask_id="subtask_10",
            formatter_desc=cot_agent_desc10f
        )
        agents.append(f"SpecificFormat agent {results10f['formatter_agent'].id}, formatting false branch final solution, thinking: {results10f['thinking'].content}; answer: {results10f['answer'].content}")
        sub_tasks.append(f"Sub-task 10 output: thinking - {results10f['thinking'].content}; answer - {results10f['answer'].content}")
        logs.append(results10f['subtask_desc'])

    if condition:
        final_thinking = results10['thinking']
        final_answer = results10['answer']
    else:
        final_thinking = results10f['thinking']
        final_answer = results10f['answer']

    final_result = await self.make_final_answer(final_thinking, final_answer, sub_tasks, agents)
    return final_result, logs
