async def forward_178(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize the given matrices W, X, Y, Z, including their dimensions, complex entries, and formatting. "
            "Provide a clear, unambiguous representation suitable for further analysis. Input content are results (both thinking and answer) from: none."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_sc_instruction_0_2 = (
            "Sub-task 2: Rigorously verify and record key matrix properties for W, X, Y, Z: check Hermiticity (A†=A), skew-Hermiticity (A†=-A), and unitarity (U†U=I) where applicable. "
            "For matrix X, explicitly compute its conjugate transpose and confirm whether X† = -X holds with detailed step-by-step reasoning and numerical/symbolic confirmation. Use multiple independent agents to cross-validate results to ensure consistency and prevent errors in fundamental property verification. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_0.subtask_2, respectively."
        )
        final_decision_instruction_0_2 = (
            "Sub-task 2: Synthesize and choose the most consistent answer for matrix property verification based on previous results."
        )
        cot_sc_desc_0_2 = {
            'instruction': cot_sc_instruction_0_2,
            'final_decision_instruction': final_decision_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.5,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_sc_desc_0_2,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_reflect_instruction_0_3 = (
            "Sub-task 3: Using the verified properties from stage_0.subtask_2, analyze whether the matrix exponential e^X is unitary and whether it preserves the norm of arbitrary vectors. "
            "Explicitly justify the conclusion based on the skew-Hermiticity of X and properties of matrix exponentials. Avoid recomputing X's properties; rely solely on the authoritative verification. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & former iterations of stage_0.subtask_3, respectively."
        )
        critic_instruction_0_3 = (
            "Please review and provide the limitations of provided solutions of the unitarity and norm preservation analysis of e^X."
        )
        cot_reflect_desc_0_3 = {
            'instruction': cot_reflect_instruction_0_3,
            'critic_instruction': critic_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_0_3, log_0_3 = await self.reflexion(
            subtask_id='stage_0.subtask_3',
            reflect_desc=cot_reflect_desc_0_3,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_reflect_instruction_0_4 = (
            "Sub-task 4: Examine the expression (e^X)*Y*(e^{-X}) to determine if it represents a valid quantum state (density matrix). "
            "Check Hermiticity, positivity, and trace preservation, explicitly referencing the unitarity of e^X from stage_0.subtask_3 and the properties of Y from stage_0.subtask_2. "
            "Ensure no re-derivation of X's properties occurs here to avoid inconsistency. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3 & stage_0.subtask_2 & former iterations of stage_0.subtask_4, respectively."
        )
        critic_instruction_0_4 = (
            "Please review and provide the limitations of provided solutions of the quantum state validity analysis of (e^X)*Y*(e^{-X})."
        )
        cot_reflect_desc_0_4 = {
            'instruction': cot_reflect_instruction_0_4,
            'critic_instruction': critic_instruction_0_4,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_0_4, log_0_4 = await self.reflexion(
            subtask_id='stage_0.subtask_4',
            reflect_desc=cot_reflect_desc_0_4,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_reflect_instruction_0_5 = (
            "Sub-task 5: Check if matrices Z and X are Hermitian to assess if they can represent observables. "
            "Use the verified Hermiticity results from stage_0.subtask_2 as the authoritative source. Avoid recomputing or contradicting previous results. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & former iterations of stage_0.subtask_5, respectively."
        )
        critic_instruction_0_5 = (
            "Please review and provide the limitations of provided solutions of Hermiticity check for Z and X."
        )
        cot_reflect_desc_0_5 = {
            'instruction': cot_reflect_instruction_0_5,
            'critic_instruction': critic_instruction_0_5,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_0_5, log_0_5 = await self.reflexion(
            subtask_id='stage_0.subtask_5',
            reflect_desc=cot_reflect_desc_0_5,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    cot_debate_instruction_1_1 = (
        "Sub-task 1: Evaluate the four candidate statements based on the refined and consistent analyses from stage_0 subtasks. "
        "Use Debate collaboration to allow multiple agents to discuss and challenge each other's reasoning, ensuring the final evaluation is robust and accounts for all verified matrix properties. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_0.subtask_3 & stage_0.subtask_4 & stage_0.subtask_5, respectively."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent evaluation of candidate statements based on debate results."
    )
    debate_desc_1_1 = {
        'instruction': cot_debate_instruction_1_1,
        'final_decision_instruction': final_decision_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    aggregate_instruction_1_2 = (
        "Sub-task 2: Aggregate the evaluations from the debate to select the most consistent and supported candidate statement. "
        "Ensure the aggregation respects the authoritative matrix property verifications and the logical implications derived from them. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    aggregate_desc_1_2 = {
        'instruction': aggregate_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.aggregate(
        subtask_id='stage_1.subtask_2',
        aggregate_desc=aggregate_desc_1_2
    )
    logs.append(log_1_2)

    review_instruction_2_1 = (
        "Sub-task 1: Perform a consistency check to verify that all key matrix properties (especially X's skew-Hermiticity and e^X's unitarity) used in the final answer selection are consistent with the authoritative verification in stage_0.subtask_2. "
        "Detect and flag any contradictions or re-derivations that conflict with the single source of truth to prevent errors like those in the previous attempt. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_2, respectively."
    )
    review_desc_2_1 = {
        'instruction': review_instruction_2_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + [results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Perform a final chain-of-thought reasoning to confirm the validity of the selected statement, incorporating the consistency check results. "
        "Produce a final assessment that explicitly references the verified matrix properties and logical deductions, ensuring correctness and compliance with quantum mechanics principles. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    cot_agent_desc_2_2 = {
        'instruction': cot_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])
    return final_answer, logs
