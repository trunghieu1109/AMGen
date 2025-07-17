async def forward_176(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given quantitative and qualitative information from the problem, "
        "including radius ratio, mass ratio, peak wavelength equality, radial velocities, and black body assumptions."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0_1, log0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log0_1)

    cot_sc_instruction2 = (
        "Sub-task 2: Apply physical laws relevant to the problem: Wien's displacement law to relate peak wavelength and temperature, "
        "Doppler effect to analyze the impact of radial velocities on observed wavelengths, and black body radiation formula for luminosity, "
        "based on the output from Sub-task 1."
    )
    N = self.max_sc
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results0_1.get('thinking', ''), results0_1.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results0_2, log0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=N
    )
    logs.append(log0_2)

    cot_sc_instruction3 = (
        "Sub-task 1: Combine the radius ratio and temperature equality (from peak wavelength equality) to determine the direct relationship affecting luminosity ratio, "
        "considering Doppler shift corrections if necessary, based on the output from stage_0.subtask_2."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results0_2.get('thinking', ''), results0_2.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=N
    )
    logs.append(log1_1)

    cot_sc_instruction4 = (
        "Sub-task 1: Derive the luminosity ratio L1/L2 using the formula L = 4πR^2σT^4, substituting the known radius ratio and equal temperature, "
        "and calculate the numerical factor, based on the output from stage_1.subtask_1."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results1_1.get('thinking', ''), results1_1.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=N
    )
    logs.append(log2_1)

    debate_instruction5 = (
        "Sub-task 1: Evaluate the computed luminosity ratio against the provided answer choices (~2.23, ~2.25, ~2.32, ~2.35) "
        "and select the closest matching option, based on the output from stage_2.subtask_1."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        'input': [taskInfo, results2_1.get('thinking', ''), results2_1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3_1, log3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log3_1)

    final_answer = await self.make_final_answer(results3_1.get('thinking', ''), results3_1.get('answer', ''))
    return final_answer, logs
