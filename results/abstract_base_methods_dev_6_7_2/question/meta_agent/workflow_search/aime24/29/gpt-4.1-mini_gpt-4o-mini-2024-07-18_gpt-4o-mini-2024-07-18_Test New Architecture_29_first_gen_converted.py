async def forward_29(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Select all valid chip placements on the 5x5 grid satisfying the constraints: each cell at most one chip, all chips in the same row and column have the same color, and maximality condition; simplify any ratio and compute sum of numerator and denominator if applicable."
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
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, selecting valid chip placements, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Derive a quantitative measure expressing the number of valid configurations from Subtask 1, applying transformations to express relationships clearly."
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, deriving quantitative measure, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Subtask 3: Evaluate the quantity of valid chip configurations, verify extremal values or optimality, and confirm correctness based on previous subtasks."
    critic_instruction3 = "Please review the evaluation of valid chip configurations and provide any limitations or corrections needed."
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, evaluating valid configurations, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    results4_list = []
    for i in range(self.max_sc):
        cot_sc_instruction4 = "Subtask 4: Generate candidate outputs for valid chip configurations based on previous validated results."
        cot_sc_desc4 = {
            'instruction': cot_sc_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4 = await self.sc_cot(
            subtask_id="subtask_4",
            cot_sc_desc=cot_sc_desc4,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results4['cot_agent'][0].id}, generating candidate outputs, thinking: {results4['list_thinking'][0]}; answer: {results4['list_answer'][0]}")
        results4_list.append(results4)
        sub_tasks.append(f"Subtask 4 output iteration {i+1}: thinking - {results4['list_thinking'][0]}; answer - {results4['list_answer'][0]}")
        logs.append(results4['subtask_desc'])

    aggregate_instruction5 = "Subtask 5: Aggregate candidate outputs from Subtask 4 to produce a single coherent and consistent output."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + [r['answer'] for r in results4_list],
        'temperature': 0.0,
        'context': ["user query", "candidate outputs from subtask 4"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, consolidating candidate outputs, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    condition = True

    if condition:
        # True branch
        programmer_instruction6 = "Subtask 6: Validate and verify the aggregated output against criteria, confirm correctness and completeness, and generate supplementary content if needed."
        programmer_desc6 = {
            'instruction': programmer_instruction6,
            'input': [taskInfo, results5['thinking'], results5['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
            'entry_point': "validate_and_assess"
        }
        results6 = await self.programmer(
            subtask_id="subtask_6",
            programmer_desc=programmer_desc6
        )
        agents.append(f"Programmer agent {results6['programmer_agent'].id}, validating output, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}; exec_result: {results6['exec_result']}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}; output - {results6['exec_result']}")
        logs.append(results6['subtask_desc'])

        review_instruction7 = "Subtask 7: Review and refine the validated output to ensure accuracy, consistency, and completeness."
        review_desc7 = {
            'instruction': review_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results7 = await self.review(
            subtask_id="subtask_7",
            review_desc=review_desc7
        )
        agents.append(f"Review agent {results7['review_agent'].id}, reviewing output, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        formatter_instruction8 = "Subtask 8: Enhance clarity and coherence of the reviewed output for presentation."
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "feedback of subtask 7", "correct of subtask 7"]
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, enhancing clarity, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        formatter_instruction9 = "Subtask 9: Format the final artifact to meet predefined output requirements."
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=formatter_desc9
        )
        agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, formatting artifact, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

    else:
        # False branch
        formatter_instruction6 = "Subtask 6: Identify, clarify, and validate discrete units of information from the source, ensuring correctness and relevance."
        formatter_desc6 = {
            'instruction': formatter_instruction6,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results6 = await self.specific_format(
            subtask_id="subtask_6",
            formatter_desc=formatter_desc6
        )
        agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, identifying and validating units, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

        cot_agent_instruction7 = "Subtask 7: Evaluate and validate the entity against defined criteria to confirm accuracy and completeness."
        cot_agent_desc7 = {
            'instruction': cot_agent_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results7 = await self.review(
            subtask_id="subtask_7",
            review_desc=cot_agent_desc7
        )
        agents.append(f"Review agent {results7['review_agent'].id}, evaluating entity, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        formatter_instruction8 = "Subtask 8: Refine and restructure information to enhance clarity and coherence."
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "feedback of subtask 7", "correct of subtask 7"]
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, refining clarity, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        formatter_instruction9 = "Subtask 9: Format the verified artifact to meet output requirements."
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=formatter_desc9
        )
        agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, formatting artifact, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
