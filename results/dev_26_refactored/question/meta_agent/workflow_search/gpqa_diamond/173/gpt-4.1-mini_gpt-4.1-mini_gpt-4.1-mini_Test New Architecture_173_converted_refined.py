async def forward_173(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and transform the given physical parameters into explicit numerical values and symbolic expressions suitable for further calculations. "
        "Determine the initial nucleus rest mass M (from 300 GeV rest-mass energy), the rest masses of the two fragments based on the 2:1 mass ratio and the 99% total rest mass after fission, "
        "and the total available kinetic energy from the mass defect. Ensure clarity and precision to avoid ambiguity in subsequent calculations."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent and precise expressions and values for physical parameters.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Solve the relativistic energy-momentum conservation equation exactly to find the common fragment momentum p. "
        "Specifically, numerically solve the equation: sqrt(m1^2 c^4 + p^2 c^2) - m1 c^2 + sqrt(m2^2 c^4 + p^2 c^2) - m2 c^2 = total kinetic energy (mass defect energy). "
        "Use the physical parameters extracted in Sub-task 1. This step addresses the critical failure in previous attempts where p was approximated classically, leading to incorrect kinetic energy differences. "
        "The solution must be precise and carried forward_173 for kinetic energy calculations."
    )
    final_decision_instruction2 = "Sub-task 2: Provide the exact numerical value of the fragment momentum p consistent with relativistic conservation laws."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3a = (
        "Sub-task 3a: Calculate the relativistic kinetic energy T1 of the heavier fragment using the exact momentum p obtained from stage_1.subtask_2 and the relativistic energy-momentum relation: "
        "T1_rel = sqrt(m1^2 c^4 + p^2 c^2) - m1 c^2. Perform explicit numerical evaluation to avoid qualitative or heuristic errors."
    )
    final_decision_instruction3a = "Sub-task 3a: Provide the precise numerical value of the relativistic kinetic energy T1_rel of the heavier fragment."
    debate_desc3a = {
        'instruction': debate_instruction3a,
        'final_decision_instruction': final_decision_instruction3a,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results3a, log3a = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3a,
        n_repeat=self.max_round
    )
    logs.append(log3a)

    debate_instruction3b = (
        "Sub-task 3b: Calculate the classical (non-relativistic) kinetic energy approximation T1_classic of the heavier fragment using the same momentum p and the classical formula T1_classic = p^2 / (2 m1). "
        "This ensures a consistent basis for comparison and avoids the previous error of using different momenta for relativistic and classical calculations."
    )
    final_decision_instruction3b = "Sub-task 3b: Provide the precise numerical value of the classical kinetic energy approximation T1_classic of the heavier fragment."
    debate_desc3b = {
        'instruction': debate_instruction3b,
        'final_decision_instruction': final_decision_instruction3b,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results3b, log3b = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc3b,
        n_repeat=self.max_round
    )
    logs.append(log3b)

    debate_instruction4 = (
        "Sub-task 4: Compute the absolute difference between the relativistic kinetic energy T1_rel and the classical kinetic energy approximation T1_classic for the heavier fragment. "
        "Then, select the closest answer choice from the given options: 10 MeV, 5 MeV, 2 MeV, or 20 MeV. "
        "Explicitly verify the magnitude of relativistic corrections by comparing numerical results, preventing assumption-driven errors."
    )
    final_decision_instruction4 = "Sub-task 4: Provide the final answer choice corresponding to the absolute difference between relativistic and classical kinetic energies."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3a['thinking'], results3a['answer'], results3b['thinking'], results3b['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
