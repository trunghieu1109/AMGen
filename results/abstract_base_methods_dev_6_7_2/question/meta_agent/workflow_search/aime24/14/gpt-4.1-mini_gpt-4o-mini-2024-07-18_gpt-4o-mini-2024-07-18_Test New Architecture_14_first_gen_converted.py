async def forward_14(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_2 = "Sub-task 2: Formulate the equation(s) relating hyperbola parameters of opposite vertices so that the distances between adjacent points are equal (rhombus condition)."
    debate_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc_2,
        final_decision_desc={
            'instruction': "Sub-task 2: Make final decision on the formulated equations and constraints.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, formulating rhombus condition equations, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding on rhombus condition equations, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_1 = "Sub-task 1: Parameterize points on x²/20–y²/24=1, derive BD² expression and side-equality constraints using these parameters, based on the rhombus condition equations from Sub-task 2."
    cot_sc_desc_1 = {
        'instruction': cot_sc_instruction_1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc_1,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, parameterizing points and deriving BD², thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_0 = "Sub-task 0: Identify and extract the defining features: hyperbola equation, rhombus properties, and origin-intersection condition of diagonals, using outputs from Sub-task 1."
    cot_agent_desc_0 = {
        'instruction': cot_instruction_0,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results0 = await self.cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_agent_desc_0
    )
    agents.append(f"CoT agent {results0['cot_agent'].id}, extracting defining features, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    candidates = []
    for i in range(10):
        answergen_instruction_3 = f"Sub-task 3: Iteratively generate candidate parameter pairs (t1,t2) for iteration {i}, compute BD² and verify rhombus side equality, documenting each computation."
        answergen_desc_3 = {
            'instruction': answergen_instruction_3,
            'input': [taskInfo, results0['thinking'], results0['answer']],
            'temperature': 0.3,
            'context': ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results3 = await self.answer_generate(
            subtask_id=f"subtask_3_{i}",
            cot_agent_desc=answergen_desc_3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, generating candidate parameters iteration {i}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Sub-task 3 iteration {i} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        candidates.append(results3['answer'].content)

    aggregate_instruction_4 = "Sub-task 4: Aggregate all valid BD² values from candidates, evaluate consistency, select the highest lower-bound (supremum) below computed values, and validate correctness."
    aggregate_desc_4 = {
        'instruction': aggregate_instruction_4,
        'input': [taskInfo] + candidates,
        'temperature': 0.0,
        'context': ["user query", "candidate BD² values"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc_4
    )
    agents.append(f"CoT agent {results4['aggregate_agent'].id}, aggregating candidate BD² values, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    condition = 'gaps' in results4['answer'].content.lower() or 'ambiguity' in results4['answer'].content.lower()

    if condition:
        formatter_instruction_6 = "Sub-task 6: Validate the final supremum BD² against an analytical derivation or known bound to confirm accuracy and completeness."
        review_desc_6 = {
            'instruction': formatter_instruction_6,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results6 = await self.review(
            subtask_id="subtask_6",
            review_desc=review_desc_6
        )
        agents.append(f"Review agent {results6['review_agent'].id}, validating supremum BD², feedback: {results6['thinking'].content}; correct: {results6['answer'].content}")
        sub_tasks.append(f"Sub-task 6 output: feedback - {results6['thinking'].content}; correct - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

        formatter_instruction_7 = "Sub-task 7: Refine the presentation of the derived supremum to enhance clarity, coherence, and logical flow in the final result."
        formatter_desc_7 = {
            'instruction': formatter_instruction_7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "feedback of subtask 6", "correct of subtask 6"]
        }
        results7 = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=formatter_desc_7
        )
        agents.append(f"CoT agent {results7['formatter_agent'].id}, refining clarity and coherence, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        formatter_instruction_8 = "Sub-task 8: Format the validated result into the specified JSON integer-only output structure."
        formatter_desc_8 = {
            'instruction': formatter_instruction_8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
            'format': 'short and concise, without explanation'
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc_8
        )
        agents.append(f"CoT agent {results8['formatter_agent'].id}, formatting final output, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
        return final_answer, logs

    else:
        formatter_instruction_5 = "Sub-task 5: Identify any discrete gaps or ambiguities in the numeric/analytic search, clarify causes of missing bounds, and validate that each failure mode meets criteria for further refinement."
        answergen_desc_5 = {
            'instruction': formatter_instruction_5,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results5 = await self.answer_generate(
            subtask_id="subtask_5",
            cot_agent_desc=answergen_desc_5
        )
        agents.append(f"CoT agent {results5['cot_agent'].id}, identifying gaps and ambiguities, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])

        final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
        return final_answer, logs

    # End of sequential flow
