async def forward_159(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Extract and explicitly list all given quantitative and qualitative information from the query, including aperture geometry (N-sided polygon with apothem a), light properties (monochromatic wavelength λ, propagation direction), limit condition (N → ∞), and approximations (small-angle tanθ ≈ θ). This step must avoid missing any parameter to ensure consistent context for subsequent subtasks."
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

    debate_instruction2 = "Sub-task 2: Analyze and clarify the geometric relationship between the polygon aperture and its limiting circular aperture as N → ∞. Explicitly distinguish the apothem (inradius) from the circumradius (radius of the limiting circle). Derive the formula relating apothem a and circumradius R (R = a / cos(π/N)) and take the limit as N → ∞ to find the correct radius R to use in diffraction calculations. This step must correct the previous error of equating apothem to radius and ensure the corrected radius is passed forward_159."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", results1['thinking'], results1['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Derive the angular positions of the diffraction minima for a circular aperture of radius R (corrected circumradius) illuminated by monochromatic light of wavelength λ, using the small-angle approximation. Use the known zeros of the Bessel function J1 to express minima positions and formulate the angular distance between the first two minima. Ensure the derivation explicitly uses the corrected radius from Subtask 2 to avoid geometric misinterpretation."
    cot_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer']]
    }
    results3, log3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_desc3
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Calculate the numerical value of the angular distance between the first two minima using the derived formula and the corrected radius. Then, explicitly compare this numeric result against each given choice by listing each choice's value and computing the absolute difference. Select the choice with the smallest absolute difference to avoid mislabeling errors. This step must incorporate the feedback to prevent confusion between close numeric values and their corresponding choice letters."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
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
