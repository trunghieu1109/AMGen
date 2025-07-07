async def forward_10(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction7 = "Sub-task 7: Plan the derivation of CE using rectangle properties and power of a point or circle theorems."
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results7 = await self.sc_cot(
        subtask_id="subtask_7",
        cot_sc_desc=cot_agent_desc7,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results7['list_thinking']):
        agents.append(f"CoT-SC agent {results7['cot_agent'][idx].id}, planning derivation of CE, thinking: {results7['list_thinking'][idx]}; answer: {results7['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction1 = "Sub-task 1: Extract and list all essential geometric elements and relationships (rectangles, circle, collinear points, given lengths) from the problem statement."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting geometric elements, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Sub-task 2: Generate potential equations or geometric constructions (e.g., power of point, similar triangles) to relate CE to known lengths."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    candidate_equations = []
    for i in range(self.max_sc):
        results2 = await self.sc_cot(
            subtask_id="subtask_2",
            cot_sc_desc=cot_sc_desc2,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results2['cot_agent'][0].id}, generating candidate equations, thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        candidate_equations.append((results2['thinking'], results2['answer']))

    aggregate_instruction3 = "Sub-task 3: Integrate all candidate relations into a unified equation for CE."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + candidate_equations,
        'temperature': 0.0,
        'context': ["user query", "candidate equations from subtask 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, consolidating candidate equations, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_sc_instruction1_1 = "Sub-task 1 (part 2): Evaluate the given numeric inputs (AB, BC, EF, FG) and characterize their roles in the rectangle and circle configuration."
    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    results1_1 = await self.sc_cot(
        subtask_id="subtask_1_1",
        cot_sc_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results1_1['list_thinking']):
        agents.append(f"CoT-SC agent {results1_1['cot_agent'][idx].id}, evaluating numeric inputs, thinking: {results1_1['list_thinking'][idx]}; answer: {results1_1['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 1 (part 2) output: thinking - {results1_1['thinking'].content}; answer - {results1_1['answer'].content}")
    logs.append(results1_1['subtask_desc'])

    review_instruction4 = "Sub-task 4: Validate the derived equation against the rectangle and circle constraints for accuracy and consistency."
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
    agents.append(f"Review agent {results4['review_agent'].id}, validating derived equation, feedback: {results4['thinking'].content}; correct: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: feedback - {results4['thinking'].content}; correct - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    revise_instruction5 = "Sub-task 5: Refine the reasoning steps and outputs to enhance clarity and coherence."
    revise_desc5 = {
        'instruction': revise_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "feedback of subtask 4", "correct of subtask 4"]
    }
    results5 = await self.revise(
        subtask_id="subtask_5",
        revise_desc=revise_desc5
    )
    agents.append(f"Revise agent {results5['revise_agent'].id}, refining solution, thinking: {results5['thinking'].content}; revised_solution: {results5['revised_solution'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; revised_solution - {results5['revised_solution'].content}")
    logs.append(results5['subtask_desc'])

    formatter_instruction6 = "Sub-task 6: Format the validated solution steps into the required presentation structure."
    formatter_desc6 = {
        'instruction': formatter_instruction6,
        'input': [taskInfo, results5['thinking'], results5['revised_solution']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "revised solution of subtask 5"],
        'format': 'short and concise, without explanation'
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=formatter_desc6
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, formatting solution, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    if "units" in results6['answer'].content.lower() or "ambiguous" in results6['answer'].content.lower():
        formatter_instruction8 = "Sub-task 8: Identify and clarify any remaining units or ambiguous measures in the construction, and validate they meet problem criteria."
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"],
            'format': 'short and concise, without explanation'
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, clarifying units, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        answer_generate_instruction8 = "Sub-task 8: Generate final answer after unit clarification."
        answer_generate_desc8 = {
            'instruction': answer_generate_instruction8,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
        }
        results8b = await self.answer_generate(
            subtask_id="subtask_8b",
            cot_agent_desc=answer_generate_desc8
        )
        agents.append(f"AnswerGenerate agent {results8b['cot_agent'].id}, generating final answer, thinking: {results8b['thinking'].content}; answer: {results8b['answer'].content}")
        sub_tasks.append(f"Sub-task 8b output: thinking - {results8b['thinking'].content}; answer - {results8b['answer'].content}")
        logs.append(results8b['subtask_desc'])

        final_answer = await self.make_final_answer(results8b['thinking'], results8b['answer'], sub_tasks, agents)
        return final_answer, logs
    else:
        review_instruction9 = "Sub-task 9: Review the final JSON structure for completeness, consistency, and adherence to the control flow template."
        review_desc9 = {
            'instruction': review_instruction9,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results9 = await self.review(
            subtask_id="subtask_9",
            review_desc=review_desc9
        )
        agents.append(f"Review agent {results9['review_agent'].id}, reviewing final JSON, feedback: {results9['thinking'].content}; correct: {results9['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: feedback - {results9['thinking'].content}; correct - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        formatter_instruction10 = "Sub-task 10: Refine and polish the JSON output presentation to ensure clarity and strict template compliance."
        formatter_desc10 = {
            'instruction': formatter_instruction10,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"],
            'format': 'short and concise, without explanation'
        }
        results10 = await self.specific_format(
            subtask_id="subtask_10",
            formatter_desc=formatter_desc10
        )
        agents.append(f"SpecificFormat agent {results10['formatter_agent'].id}, polishing JSON output, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
        sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

        final_answer = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents)
        return final_answer, logs
