async def forward_19(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Control Flow 0: start sequential

    # Stage 2: subtask_2_select_element_by_conformity_evaluation
    debate_instruction = "Sub-task 1: Evaluate the product's factors by examining the expression 2 - 2ω^k + ω^(2k) for roots of unity, identify patterns, and determine appropriate evaluation criteria."
    debate_desc = {
        'instruction': debate_instruction,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc,
        final_decision_desc={
            'instruction': "Sub-task 1: Make final decision on evaluation of product factors.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results1['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating product factors, thinking: {results1['list_thinking'][round][idx].content}; answer: {results1['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, evaluating product factors, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    # Stage 1: subtask_1_quantitative_data_integration
    cot_reflect_instruction = "Sub-task 2: Combine numeric insights from factor evaluations to integrate into a unified expression reflecting the entire product."
    critic_instruction = "Please review the combined numeric insights and provide limitations or improvements."
    cot_reflect_desc = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, combining numeric insights, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    # Stage 0: subtask_0_apply_transformation
    cot_sc_instruction = "Sub-task 3: Apply algebraic transformations to simplify each term 2 - 2ω^k + ω^{2k} using properties of roots of unity."
    N = self.max_sc
    cot_sc_desc = {
        'instruction': cot_sc_instruction,
        'input': [taskInfo, results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc,
        n_repeat=N
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, applying algebraic transformations, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    # Control Flow 1: start loop over k=0 to 12
    candidate_expressions = []
    for k in range(13):
        cot_instruction = f"Sub-task 4: Generate candidate expression for factor with k={k} by applying simplified form."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo, results3['thinking'], results3['answer'], k],
            'temperature': 0.0,
            'context': ["user query", f"simplified form from subtask 3", f"k={k}"]
        }
        results4 = await self.cot(
            subtask_id=f"subtask_4_{k+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results4['cot_agent'].id}, generating candidate for k={k}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4_{k+1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        candidate_expressions.append((results4['thinking'], results4['answer']))

    # Stage 4: subtask_4_consolidate_multiple_inputs
    aggregate_instruction = "Sub-task 5: Aggregate candidate expressions across all k to derive a cohesive closed-form product."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_expressions,
        'temperature': 0.0,
        'context': ["user query", "candidate expressions from subtask 4"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"CoT agent {results5['aggregate_agent'].id}, aggregating candidates, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    # Stage 5: subtask_5_identify_and_fill_gaps
    debate_instruction_5 = "Sub-task 6: Identify deficiencies in the aggregated derivation and fill any missing algebraic justifications or computational steps."
    debate_desc_5 = {
        'instruction': debate_instruction_5,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    final_decision_desc_5 = {
        'instruction': "Sub-task 6: Make final decision on completeness and correctness of derivation.",
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc_5,
        final_decision_desc=final_decision_desc_5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results6['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, filling gaps, thinking: {results6['list_thinking'][round][idx].content}; answer: {results6['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, filling gaps, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    # Control Flow 3: start conditional
    # Condition: check if the derivation directly yields an integer result
    condition_met = False
    try:
        # Simple heuristic: if answer content contains an integer or numeric result
        int_val = int(results6['answer'].content.strip())
        condition_met = True
    except:
        condition_met = False

    if condition_met:
        # Control Flow 4: start true branch
        # Directly valid integer result, proceed without additional unit checks
        # Control Flow 5: end true branch
        final_thinking = results6['thinking']
        final_answer = results6['answer']
    else:
        # Control Flow 6: start false branch
        # Perform discrete unit checks
        # Stage 7: subtask_7_enhance_clarity_and_coherence
        formatter_instruction = "Sub-task 7: Enhance clarity and coherence of the overall derivation and explanation."
        formatter_desc = {
            'instruction': formatter_instruction,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 6", "answer of subtask 6"],
            'format': 'short and concise, without explaination'
        }
        results7 = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=formatter_desc
        )
        agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, enhancing clarity, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        # Stage 8: subtask_8_validate_and_assess
        review_instruction = "Sub-task 8: Validate the final result by assessing its consistency, correctness, and completeness."
        review_desc = {
            'instruction': review_instruction,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
        }
        results8 = await self.review(
            subtask_id="subtask_8",
            review_desc=review_desc
        )
        agents.append(f"Review agent {results8['review_agent'].id}, validating result, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        # Stage 6: subtask_6_identify_clarify_validate_units
        answer_generate_instruction = "Sub-task 9: Identify, clarify, and validate discrete reasoning units ensuring correctness criteria."
        answer_generate_desc = {
            'instruction': answer_generate_instruction,
            'input': [taskInfo, results8['thinking'], results8['answer'], results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8", "thinking of subtask 7", "answer of subtask 7"]
        }
        results9 = await self.answer_generate(
            subtask_id="subtask_9",
            cot_agent_desc=answer_generate_desc
        )
        agents.append(f"AnswerGenerate agent {results9['cot_agent'].id}, validating units, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        final_thinking = results9['thinking']
        final_answer = results9['answer']
        # Control Flow 7: end false branch

    # Control Flow 8: end conditional

    # Stage 9: subtask_9_format_artifact
    formatter_instruction_final = "Sub-task 10: Format the final remainder result as an integer modulo 1000 in the specified output format."
    formatter_desc_final = {
        'instruction': formatter_instruction_final,
        'input': [taskInfo, final_thinking, final_answer],
        'temperature': 0.0,
        'context': ["user query", "final thinking", "final answer"],
        'format': 'short and concise, without explaination'
    }
    results10 = await self.specific_format(
        subtask_id="subtask_10",
        formatter_desc=formatter_desc_final
    )
    agents.append(f"SpecificFormat agent {results10['formatter_agent'].id}, formatting final answer, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
    logs.append(results10['subtask_desc'])

    final_answer_processed = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents)
    # Control Flow 9: end sequential
    return final_answer_processed, logs
