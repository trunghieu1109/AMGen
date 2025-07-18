async def forward_158(self, taskInfo):
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Classify the nature of the observed spectral feature at 790 nm by explicitly considering "
        "whether it is an emission line or an absorption break (such as the Lyman limit at 912 Å or Lyα forest at 1216 Å). "
        "Embed the failure reason from previous attempts that misinterpreted the feature as an emission line, leading to incorrect redshift estimates. "
        "Use self-consistent chain-of-thought reasoning to analyze the spectral shape, especially the sharp flux drop at wavelengths shorter than 790 nm, to correctly identify the feature type."
    )

    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }

    results_1_1, log_1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Derive the redshift of the quasar by relating the observed 790 nm feature to possible rest-frame wavelengths under both emission-line and absorption-break interpretations. "
        "Explicitly compute redshifts for each scenario (e.g., z_emission = (lambda_obs / lambda_emitted) - 1 and z_absorption = (lambda_obs / lambda_absorption) - 1), then evaluate which redshift best matches the observed spectral characteristics, including the flux drop. "
        "Avoid the previous error of locking into emission-line redshifts without considering absorption breaks. Justify the chosen redshift with clear reasoning."
    )

    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }

    results_1_2, log_1_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=cot_agent_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Compute the comoving distance to the quasar using the confirmed redshift from stage_1.subtask_2 and the given Lambda-CDM cosmological parameters "
        "(H_0 = 70 km/s/Mpc, Omega_m = 0.3, Omega_Lambda = 0.7, flat universe). Perform precise numerical integration of c/H(z) over redshift to obtain the comoving distance at scale factor a=1 (present epoch). "
        "Embed feedback to ensure numerical precision and clarity in the calculation, avoiding ambiguity in the results."
    )

    cot_sc_desc_2_1 = {
        'instruction': cot_sc_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }

    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc_2_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Select the most appropriate discrete comoving distance value from the provided choices (6, 7, 8, 9 Gpc) by quantitatively comparing the computed continuous comoving distance from stage_2.subtask_1. "
        "Calculate the numerical proximity of the computed value to each choice and select the closest one, rather than relying on majority vote or qualitative reasoning. "
        "Use a structured collaborative chain-of-thought pattern to reconcile any differences and reach a consensus based on numerical accuracy, embedding the previous failure reason where the final choice did not reflect the best numerical match."
    )

    cot_sc_desc_3_1 = {
        'instruction': cot_sc_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }

    results_3_1, log_3_1 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc_3_1,
        n_repeat=self.max_sc
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
