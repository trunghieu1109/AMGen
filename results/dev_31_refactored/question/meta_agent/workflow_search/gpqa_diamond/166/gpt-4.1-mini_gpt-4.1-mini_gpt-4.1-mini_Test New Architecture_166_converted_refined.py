async def forward_166(self, taskInfo):
    logs = []
    loop_results = {}
    loop_results['stage_0.subtask_1'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_2'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_3'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_4'] = {'thinking': [], 'answer': []}
    loop_results['stage_1.subtask_1'] = {'thinking': [], 'answer': []}
    loop_results['stage_1.subtask_2'] = {'thinking': [], 'answer': []}
    loop_results['stage_1.subtask_3'] = {'thinking': [], 'answer': []}

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task stage_0.subtask_1: Compute the normalization constant N and construct the normalized Schrödinger cat state |psi> for given phi and alpha, "
            "ensuring correct normalization and explicit expression of the state vector. Input content: taskInfo."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task stage_0.subtask_2: Form the density matrix ρ = |psi><psi| of the non-Gaussian Schrödinger cat state using the normalized state from subtask_1. "
            "Input content: taskInfo, all thinking and answers from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task stage_0.subtask_3: Compute the first moments (mean values) and second moments (covariance matrix) of the quadrature operators for the state ρ, "
            "explicitly deriving all necessary expectation values to characterize the state’s Gaussian reference. Input content: taskInfo, all thinking and answers from stage_0.subtask_2 iterations."
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_instruction_0_4 = (
            "Sub-task stage_0.subtask_4: Construct the covariance matrix of the reference Gaussian state τ using the first and second moments computed in subtask_3, "
            "ensuring the Gaussian state matches these moments exactly as required for the relative entropy measure. Input content: taskInfo, all thinking and answers from stage_0.subtask_3 iterations."
        )
        cot_agent_desc_0_4 = {
            'instruction': cot_instruction_0_4,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_0_4, log_0_4 = await self.cot(subtask_id='stage_0.subtask_4', cot_agent_desc=cot_agent_desc_0_4)
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_instruction_1_1 = (
            "Sub-task stage_1.subtask_1: Calculate the symplectic eigenvalue(s) of the covariance matrix from subtask_4, "
            "which are necessary to compute the von Neumann entropy of the Gaussian reference state τ using the standard formula S(τ) = Σ h(ν_i), where h(ν) = (ν+½)ln(ν+½) - (ν-½)ln(ν-½). "
            "Input content: taskInfo, all thinking and answers from stage_0.subtask_4 iterations."
        )
        cot_agent_desc_1_1 = {
            'instruction': cot_instruction_1_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])
        logs.append(log_1_1)

        cot_instruction_1_2 = (
            "Sub-task stage_1.subtask_2: Compute the von Neumann entropy S(ρ) of the pure state ρ from subtask_2, confirming it is zero, to avoid any unsupported assumptions. "
            "Input content: taskInfo, all thinking and answers from stage_0.subtask_2 iterations."
        )
        cot_agent_desc_1_2 = {
            'instruction': cot_instruction_1_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_1_2, log_1_2 = await self.cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_agent_desc_1_2)
        loop_results['stage_1.subtask_2']['thinking'].append(results_1_2['thinking'])
        loop_results['stage_1.subtask_2']['answer'].append(results_1_2['answer'])
        logs.append(log_1_2)

        debate_instruction_1_3 = (
            "Sub-task stage_1.subtask_3: Calculate the relative entropy measure δ_b = trace(ρ ln ρ) - trace(τ ln τ) = -S(τ) since S(ρ) = 0, "
            "using the entropy values from subtasks 1 and 2. This subtask uses a Debate agent collaboration pattern with two independent agents computing S(τ) and cross-checking results to ensure correctness and avoid the previous failure of unverified entropy values. "
            "Input content: taskInfo, all thinking and answers from stage_1.subtask_1 and stage_1.subtask_2 iterations."
        )
        final_decision_instruction_1_3 = (
            "Sub-task stage_1.subtask_3: Calculate the relative entropy measure δ_b = trace(ρ ln ρ) - trace(τ ln τ) = -S(τ) since S(ρ) = 0, "
            "using the entropy values from subtasks 1 and 2, and finalize the consistent result."
        )
        debate_desc_1_3 = {
            'instruction': debate_instruction_1_3,
            'final_decision_instruction': final_decision_instruction_1_3,
            'input': [taskInfo] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'] + loop_results['stage_1.subtask_2']['thinking'] + loop_results['stage_1.subtask_2']['answer'],
            'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2'],
            'temperature': 0.5
        }
        results_1_3, log_1_3 = await self.debate(subtask_id='stage_1.subtask_3', debate_desc=debate_desc_1_3, n_repeat=self.max_round)
        loop_results['stage_1.subtask_3']['thinking'].append(results_1_3['thinking'])
        loop_results['stage_1.subtask_3']['answer'].append(results_1_3['answer'])
        logs.append(log_1_3)

    cot_formatter_instruction_3_1 = (
        "Sub-task stage_3.subtask_1: Format the final non-Gaussianity value δ_b and summarize the results, including the chosen parameters phi = -π/4 and alpha = 0.5, "
        "ensuring clarity and completeness of the output. Input content: taskInfo, all thinking and answers from stage_1.subtask_3 iterations."
    )
    formatter_desc_3_1 = {
        'instruction': cot_formatter_instruction_3_1,
        'input': [taskInfo] + loop_results['stage_1.subtask_3']['thinking'] + loop_results['stage_1.subtask_3']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3'],
        'format': 'short and concise, without explaination'
    }
    results_3_1, log_3_1 = await self.specific_format(subtask_id='stage_3.subtask_1', formatter_desc=formatter_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
