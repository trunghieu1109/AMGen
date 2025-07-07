async def forward_1(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction1 = "Subtask 1: Select all key geometric elements and configurations: triangle ABC inscribed in ω, tangents at B,C meeting at D, intersection point P of AD with ω."
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
    for idx, cot_agent in enumerate(results1['cot_agent']):
        agents.append(f"CoT-SC agent {cot_agent.id}, selecting geometric elements, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Formulate and analyze the formal relationships among AB, BC, AC, tangents, and chords to determine relations that will yield AP, based on outputs from Subtask 1."
    cot_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, analyzing formal relationships, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 0 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    candidate_relations = []
    for i in range(1):
        answer_generate_instruction = "Subtask 3: Propose an initial power-of-point relation at D: DB^2 = DC^2, or other tangent-chord relations to relate known lengths, using outputs from Subtask 0."
        answer_generate_desc = {
            'instruction': answer_generate_instruction,
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results3 = await self.answer_generate(
            subtask_id="subtask_2",
            cot_agent_desc=answer_generate_desc
        )
        agents.append(f"AnswerGenerate agent {results3['cot_agent'].id}, proposing candidate relation, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        candidate_relations.append((results3['thinking'], results3['answer']))

    aggregate_instruction = "Subtask 4: Aggregate candidate relations to derive an expression linking AD·AP with known side lengths AB·AC using power-of-point at A, based on candidate relations and further computations."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + [cr[1] for cr in candidate_relations],
        'temperature': 0.0,
        'context': ["user query", "candidate relations"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating candidate relations, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    answer_generate_instruction2 = "Subtask 5: Identify and fill computational gaps: compute AD in terms of AB, AC, BC and solve for AP, based on aggregation results."
    answer_generate_desc2 = {
        'instruction': answer_generate_instruction2,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results5 = await self.answer_generate(
        subtask_id="subtask_4",
        cot_agent_desc=answer_generate_desc2
    )
    agents.append(f"AnswerGenerate agent {results5['cot_agent'].id}, computing AD and solving for AP, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    review_instruction = "Subtask 6: Enhance clarity and coherence by restructuring the derivation into a clear, step-by-step solution, based on previous computations."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results6 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results6['review_agent'].id}, refining solution clarity, feedback: {results6['thinking'].content}; correct: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results6['thinking'].content}; correct - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    condition = False

    if condition:
        pass
    else:
        specific_format_instruction = "Subtask 7: Identify, clarify, and validate the fractional form of AP = m/n and confirm m, n are relatively prime, based on refined solution."
        specific_format_desc = {
            'instruction': specific_format_instruction,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
            'format': 'short and concise, without explaination'
        }
        results7 = await self.specific_format(
            subtask_id="subtask_6",
            formatter_desc=specific_format_desc
        )
        agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, validating fraction form, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        review_instruction2 = "Subtask 8: Validate and assess the computed fraction m/n for correctness and simplification."
        review_desc2 = {
            'instruction': review_instruction2,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
        }
        results8 = await self.review(
            subtask_id="subtask_7",
            review_desc=review_desc2
        )
        agents.append(f"Review agent {results8['review_agent'].id}, validating fraction correctness, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        specific_format_instruction2 = "Subtask 9: Format the final answer as the integer m+n according to output requirements."
        specific_format_desc2 = {
            'instruction': specific_format_instruction2,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
            'format': 'short and concise, without explaination'
        }
        results9 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=specific_format_desc2
        )
        agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, formatting final answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
