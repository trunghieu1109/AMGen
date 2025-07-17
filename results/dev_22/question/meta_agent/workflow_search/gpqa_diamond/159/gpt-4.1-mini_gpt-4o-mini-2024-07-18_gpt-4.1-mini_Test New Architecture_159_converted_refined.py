async def forward_159(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Extract and summarize all given quantitative and qualitative information about the aperture, wavelength, diffraction pattern, and approximations. Emphasize clarifying the polygonal aperture's geometry (regular polygon with equal apothems), the limit as N approaches infinity, and the small-angle approximation. This sets a clear and precise problem definition to avoid ambiguity in later steps."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze the geometric limit of the polygonal aperture as N approaches infinity, establishing that it converges to a circular aperture with radius equal to the apothem length a. Explicitly confirm the equivalence of the aperture shape and size in this limit to prepare for applying circular aperture diffraction theory."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Derive the correct expression for the angular positions of the diffraction minima for the circular aperture (limit of polygonal aperture) using Fraunhofer diffraction theory. Incorporate the Bessel function J_1 zeros that define the Airy pattern minima. Calculate the first two zeros of J_1 and express the corresponding angular minima using the small-angle approximation. Explicitly avoid using the single-slit formula and use the Airy disk formula theta_m approx x_m * lambda / (2 * pi * a), where x_m are zeros of J_1."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = "Sub-task 4: Based on the outputs from Sub-task 3, perform a consistency check by comparing the derived angular minima expressions with the well-known standard Airy pattern result (first minimum at approximately 1.22 lambda/D, with D=2a). Verify that the derived formula and numerical values align with established optics literature to prevent propagation of incorrect models."
    critic_instruction4 = "Please review the consistency check and provide any limitations or corrections needed."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Based on the output of Sub-task 4, calculate the angular distance between the first two minima using the verified expressions. Compare the calculated angular distance with the given multiple-choice options and select the correct answer. Ensure that the selection is based on the physically and mathematically consistent derivation."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
