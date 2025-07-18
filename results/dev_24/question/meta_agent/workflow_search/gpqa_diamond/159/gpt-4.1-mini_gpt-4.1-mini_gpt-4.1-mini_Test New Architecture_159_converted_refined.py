async def forward_159(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Precisely analyze the geometry of the polygonal aperture with equal apothems 'a' and rigorously derive the relationship between the apothem 'a' and the circumscribed radius R of the polygon. "
        "Emphasize that as N approaches infinity, the polygon approaches a circular aperture of radius R = a / cos(pi/N). "
        "Explicitly clarify that the aperture radius relevant for diffraction calculations is R, not a, to avoid foundational geometric errors. "
        "Use the given taskInfo for context."
    )
    debate_desc1 = {
        'instruction': cot_instruction1,
        'final_decision_instruction': "Sub-task 1: Provide a rigorous geometric relationship and conclusion for the aperture radius R.",
        'input': [taskInfo],
        'context_desc': ["user query"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Using the corrected aperture radius R from Subtask 1, analyze the far-field diffraction pattern of the circular aperture formed in the limit N → ∞. "
        "Apply the small-angle approximation (tan theta ≈ theta) and identify the angular positions of the first two minima in the Airy diffraction pattern, expressing them in terms of wavelength lambda and radius R. "
        "Ensure the correct geometric parameter R is used to avoid underestimating the angular scale. "
        "Use the thinking and answer from Subtask 1 as context."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct angular positions of the first two minima for the circular aperture diffraction pattern, based on the corrected radius R."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Calculate the angular distance Δθ between the first two minima using the Airy disk formula and the small-angle approximation, "
        "explicitly expressing the result in terms of wavelength lambda and the corrected aperture radius R. "
        "Carry forward_159 the corrected geometry from Subtask 2 and avoid errors in parameter substitution. "
        "Use the thinking and answer from Subtask 2 as context."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the explicit formula and numerical value for the angular distance Δθ between the first two minima, in terms of lambda and R."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Explicitly restate each multiple-choice option with its corresponding numeric expression. "
        "Compare the computed angular distance Δθ from Subtask 3 against each option, identify the exact match, and justify the selection of the correct answer choice. "
        "Prevent mislabeling errors by enforcing explicit mapping and reflective check between the computed value and the provided options. "
        "Use the thinking and answer from Subtask 3 as context."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the correct multiple-choice answer based on the computed angular distance Δθ and provide justification."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
