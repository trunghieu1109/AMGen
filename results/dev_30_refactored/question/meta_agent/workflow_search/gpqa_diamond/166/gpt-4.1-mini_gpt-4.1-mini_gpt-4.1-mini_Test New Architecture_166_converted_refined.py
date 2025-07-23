async def forward_166(self, taskInfo):
    logs = []
    loop_results = {}
    loop_results['stage_0.subtask_1'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_2'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_3'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_4a'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_4b'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_5'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_6'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_7'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_8'] = {'thinking': [], 'answer': []}

    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Calculate the normalization constant N for the Schrödinger cat state using the formula N = sqrt(1 + sin(2*phi)*exp(-2*alpha^2)) "
            "with the given phi and alpha values (phi = -pi/4, alpha = 0.5). This ensures the state is properly normalized before further calculations. "
            "Input content are results (both thinking and answer) from: none (initial input)."
        )
        cot_agent_desc_1 = {
            'instruction': cot_instruction_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_1, log_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_instruction_2 = (
            "Sub-task 2: Construct the normalized Schrödinger cat state |psi> = (cos(phi)|alpha> + sin(phi)|-alpha>)/N using the normalization constant from stage_0.subtask_1. "
            "This state will be used to form the density matrix. Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        cot_agent_desc_2 = {
            'instruction': cot_instruction_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_2, log_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_instruction_3 = (
            "Sub-task 3: Form the density matrix rho = |psi><psi| of the non-Gaussian Schrödinger cat state from the normalized state |psi> constructed in stage_0.subtask_2. "
            "This density matrix is the basis for moment calculations. Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
        )
        cot_agent_desc_3 = {
            'instruction': cot_instruction_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_3, log_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        cot_reflect_instruction_4a = (
            "Sub-task 4a: Explicitly compute the first moments (mean displacement vector) <x> and <p> of the Schrödinger cat state rho, carefully accounting for the non-orthogonality and overlap of coherent states. "
            "This step addresses the previous failure to calculate these moments explicitly. Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        )
        critic_instruction_4a = (
            "Please review and provide the limitations of provided solutions of first moments calculation <x> and <p> for the Schrödinger cat state rho."
        )
        cot_reflect_desc_4a = {
            'instruction': cot_reflect_instruction_4a,
            'critic_instruction': critic_instruction_4a,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_4a, log_4a = await self.reflexion(subtask_id='stage_0.subtask_4a', reflect_desc=cot_reflect_desc_4a, n_repeat=self.max_round)
        loop_results['stage_0.subtask_4a']['thinking'].append(results_4a['thinking'])
        loop_results['stage_0.subtask_4a']['answer'].append(results_4a['answer'])
        logs.append(log_4a)

        cot_reflect_instruction_4b = (
            "Sub-task 4b: Explicitly compute the second moments (covariance matrix elements) <x^2>, <p^2>, and symmetrized <xp + px> of the Schrödinger cat state rho, including interference terms due to superposition. "
            "This is critical for accurately constructing the reference Gaussian state tau and was missing in the previous attempt. Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        )
        critic_instruction_4b = (
            "Please review and provide the limitations of provided solutions of second moments calculation for the Schrödinger cat state rho."
        )
        cot_reflect_desc_4b = {
            'instruction': cot_reflect_instruction_4b,
            'critic_instruction': critic_instruction_4b,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_4b, log_4b = await self.reflexion(subtask_id='stage_0.subtask_4b', reflect_desc=cot_reflect_desc_4b, n_repeat=self.max_round)
        loop_results['stage_0.subtask_4b']['thinking'].append(results_4b['thinking'])
        loop_results['stage_0.subtask_4b']['answer'].append(results_4b['answer'])
        logs.append(log_4b)

        cot_reflect_instruction_5 = (
            "Sub-task 5: Assemble the covariance matrix sigma of the reference Gaussian state tau from the first and second moments computed in subtasks 4a and 4b. "
            "This covariance matrix fully characterizes tau and is essential for entropy calculation. Input content are results (both thinking and answer) from: stage_0.subtask_4a & stage_0.subtask_4b, respectively."
        )
        critic_instruction_5 = (
            "Please review and provide the limitations of the covariance matrix assembly for the reference Gaussian state tau."
        )
        cot_reflect_desc_5 = {
            'instruction': cot_reflect_instruction_5,
            'critic_instruction': critic_instruction_5,
            'input': [taskInfo] + loop_results['stage_0.subtask_4a']['thinking'] + loop_results['stage_0.subtask_4a']['answer'] + loop_results['stage_0.subtask_4b']['thinking'] + loop_results['stage_0.subtask_4b']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_4a', 'answer of stage_0.subtask_4a', 'thinking of stage_0.subtask_4b', 'answer of stage_0.subtask_4b']
        }
        results_5, log_5 = await self.reflexion(subtask_id='stage_0.subtask_5', reflect_desc=cot_reflect_desc_5, n_repeat=self.max_round)
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

        cot_reflect_instruction_6 = (
            "Sub-task 6: Compute the symplectic eigenvalue nu of the covariance matrix sigma of tau, using nu = sqrt(det(sigma)). "
            "This eigenvalue is required to calculate the von Neumann entropy S(tau) of the Gaussian reference state. Explicit calculation avoids assumptions and literature-based approximations. Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively."
        )
        critic_instruction_6 = (
            "Please review and provide the limitations of the symplectic eigenvalue calculation for the covariance matrix sigma."
        )
        cot_reflect_desc_6 = {
            'instruction': cot_reflect_instruction_6,
            'critic_instruction': critic_instruction_6,
            'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_6, log_6 = await self.reflexion(subtask_id='stage_0.subtask_6', reflect_desc=cot_reflect_desc_6, n_repeat=self.max_round)
        loop_results['stage_0.subtask_6']['thinking'].append(results_6['thinking'])
        loop_results['stage_0.subtask_6']['answer'].append(results_6['answer'])
        logs.append(log_6)

        cot_reflect_instruction_7 = (
            "Sub-task 7: Calculate the von Neumann entropy S(tau) of the reference Gaussian state tau using the symplectic eigenvalue nu via the formula S(tau) = (nu + 1/2) ln(nu + 1/2) - (nu - 1/2) ln(nu - 1/2). "
            "This explicit entropy calculation is necessary to correctly evaluate the relative entropy measure delta_b. Input content are results (both thinking and answer) from: stage_0.subtask_6, respectively."
        )
        critic_instruction_7 = (
            "Please review and provide the limitations of the von Neumann entropy calculation for the reference Gaussian state tau."
        )
        cot_reflect_desc_7 = {
            'instruction': cot_reflect_instruction_7,
            'critic_instruction': critic_instruction_7,
            'input': [taskInfo] + loop_results['stage_0.subtask_6']['thinking'] + loop_results['stage_0.subtask_6']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_6', 'answer of stage_0.subtask_6']
        }
        results_7, log_7 = await self.reflexion(subtask_id='stage_0.subtask_7', reflect_desc=cot_reflect_desc_7, n_repeat=self.max_round)
        loop_results['stage_0.subtask_7']['thinking'].append(results_7['thinking'])
        loop_results['stage_0.subtask_7']['answer'].append(results_7['answer'])
        logs.append(log_7)

        cot_reflect_instruction_8 = (
            "Sub-task 8: Compute the relative entropy measure delta_b = trace(rho ln rho) - trace(tau ln tau). Since rho is pure, trace(rho ln rho) = 0, so delta_b = S(tau). "
            "Use the entropy S(tau) from stage_0.subtask_7 to obtain delta_b. This step explicitly grounds the final non-Gaussianity calculation in the computed moments and entropy, avoiding previous errors of assumption. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3 & stage_0.subtask_7, respectively."
        )
        critic_instruction_8 = (
            "Please review and provide the limitations of the relative entropy measure calculation delta_b."
        )
        cot_reflect_desc_8 = {
            'instruction': cot_reflect_instruction_8,
            'critic_instruction': critic_instruction_8,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_7']['thinking'] + loop_results['stage_0.subtask_7']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_7', 'answer of stage_0.subtask_7']
        }
        results_8, log_8 = await self.reflexion(subtask_id='stage_0.subtask_8', reflect_desc=cot_reflect_desc_8, n_repeat=self.max_round)
        loop_results['stage_0.subtask_8']['thinking'].append(results_8['thinking'])
        loop_results['stage_0.subtask_8']['answer'].append(results_8['answer'])
        logs.append(log_8)

    debate_instruction_1 = (
        "Sub-task 1: Evaluate the computed non-Gaussianity values delta_b from stage_0.subtask_8 and select the best candidate value that accurately represents the non-Gaussianity for the given parameters. "
        "This selection should be based on the explicit calculations rather than literature approximations, ensuring rigor. Input content are results (both thinking and answer) from: stage_0.subtask_8, respectively."
    )
    final_decision_instruction_1 = (
        "Sub-task 1: Select the best candidate non-Gaussianity value delta_b based on explicit calculations from stage_0.subtask_8."
    )
    debate_desc_1 = {
        'instruction': debate_instruction_1,
        'final_decision_instruction': final_decision_instruction_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_8']['thinking'] + loop_results['stage_0.subtask_8']['answer'],
        'context': ['user query', 'thinking of stage_0.subtask_8', 'answer of stage_0.subtask_8'],
        'temperature': 0.5
    }
    results_9, log_9 = await self.debate(subtask_id='stage_1.subtask_1', debate_desc=debate_desc_1, n_repeat=self.max_round)
    logs.append(log_9)

    review_instruction_1 = (
        "Sub-task 1: Validate the selected non-Gaussianity value delta_b by checking its consistency with theoretical expectations, numerical stability, and parameter sensitivity for the given Schrödinger cat state parameters. "
        "This final validation ensures the solution is robust and free from previous logical gaps. Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    review_desc_1 = {
        'instruction': review_instruction_1,
        'input': [taskInfo, results_9['thinking'], results_9['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_10, log_10 = await self.review(subtask_id='stage_2.subtask_1', review_desc=review_desc_1)
    logs.append(log_10)

    final_answer = await self.make_final_answer(results_10['thinking'], results_10['answer'])
    return final_answer, logs
