async def forward_172(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and convert all given numerical data and physical constants needed for calculations, "
        "including electron mass (m), reduced Planck's constant (ħ), speed of light (c), electron speed v = 2 × 10^8 m/s, "
        "and uncertainty in position Δx = 0.1 nm = 0.1 × 10^(-9) m. Ensure these values are explicitly output as numeric constants "
        "and propagated unchanged to all subsequent subtasks to avoid the previous error of dropping or substituting the given speed."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results1, log1 = await self.sc_cot(
        subtask_id='subtask_1',
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, calculate the minimum uncertainty in momentum Δp using the Heisenberg uncertainty principle "
        "Δx Δp ≥ ħ/2, applying the exact numeric Δx from Subtask 1. Emphasize that Δp must be computed with the given Δx and that this value will be used directly in later energy uncertainty calculations."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the electron's speed relative to the speed of light by computing β = v/c and the Lorentz factor γ = 1/√(1−β²) "
        "using the exact v and c from Subtask 1. Based on this, decisively determine whether relativistic corrections are necessary (i.e., if v ≥ 0.1c). "
        "This subtask must explicitly reject the use of non-relativistic approximations if relativistic effects are significant, addressing the previous failure to apply relativistic formulas at v ≈ 0.67c."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ['user query', results1['thinking'], results1['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Derive the correct formula relating uncertainty in momentum Δp to uncertainty in kinetic energy ΔE for the electron, "
        "conditioned on the relativistic regime determined in Subtask 3. If relativistic, use the relativistic energy-momentum relation and uncertainty propagation "
        "(e.g., ΔE ≈ (p c² / E) Δp or ΔE ≈ v γ³ Δp / m). If non-relativistic, use the classical kinetic energy relation ΔE ≈ v Δp. "
        "Explicitly avoid the simplistic linear approximation without validation, embedding the feedback to prevent underestimation of ΔE."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ['user query', results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id='subtask_4',
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Numerically estimate the minimum uncertainty in energy ΔE using the formulas and values from Subtasks 2 and 4, "
        "ensuring consistent use of the given electron speed and relativistic corrections if applicable. Compare the computed ΔE with the provided multiple-choice options "
        "and select the closest value. This subtask must explicitly verify that the numerical estimate aligns with the physical regime and avoids the previous error of substituting typical atomic speeds or ignoring relativistic effects."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ['user query', results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='subtask_5',
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
