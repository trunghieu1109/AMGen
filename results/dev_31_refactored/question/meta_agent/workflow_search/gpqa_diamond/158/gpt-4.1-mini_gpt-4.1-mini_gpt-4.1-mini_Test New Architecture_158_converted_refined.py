async def forward_158(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: List plausible rest-frame spectral break candidates (e.g., Lyman limit at 91.2 nm, Ly alpha at 121.6 nm, Balmer break, etc.), "
            "compute the corresponding redshift z for each candidate using the observed 790 nm peak, and justify the most likely spectral feature causing the flux drop at wavelengths shorter than 790 nm. "
            "Input content: user query from taskInfo."
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
            "Sub-task 2: Using the justified redshift from stage_0.subtask_1 (all previous iterations), calculate the comoving distance of the quasar based on the Lambda-CDM cosmological parameters "
            "(H_0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe). Input content: user query from taskInfo, all thinking and answers from stage_0.subtask_1 iterations."
        )
        final_decision_instruction_0_2 = (
            "Sub-task 2: Synthesize and choose the most consistent comoving distance calculation based on the redshift justification from all previous iterations."
        )
        cot_sc_desc_0_2 = {
            'instruction': cot_sc_instruction_0_2,
            'final_decision_instruction': final_decision_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
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

    cot_instruction_1_1 = (
        "Sub-task 1: Combine and review the redshift justification and comoving distance calculation results from all iterations of stage_0.subtask_1 and stage_0.subtask_2 to produce a consolidated and well-justified cosmological distance estimate for the quasar. "
        "Input content: user query from taskInfo, all thinking and answers from stage_0.subtask_1 and stage_0.subtask_2 iterations."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    debate_instruction_2_1 = (
        "Sub-task 1: Evaluate the consolidated comoving distance estimate against the provided multiple-choice options (6, 7, 8, or 9 Gpc) and select the best matching value. "
        "Input content: user query from taskInfo, thinking and answer from stage_1.subtask_1."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Select the best matching comoving distance option based on the consolidated estimate."
    )
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id='stage_2.subtask_1',
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    cot_agent_instruction_3_1 = (
        "Sub-task 1: Format the selected comoving distance into the required final answer format, ensuring clarity and compliance with output specifications. "
        "Input content: user query from taskInfo, thinking and answer from stage_2.subtask_1."
    )
    cot_agent_desc_3_1 = {
        'instruction': cot_agent_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id='stage_3.subtask_1',
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
