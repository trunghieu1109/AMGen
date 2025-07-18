async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and quantify all given physical parameters and relationships from the problem statement, "
        "including initial mass M, rest-mass energy (300 GeV), fragment rest masses (m and 2m), total fragment rest mass (0.99 M), "
        "and the mass defect energy (1% of M). Ensure clarity on units and assumptions (e.g., ignoring electrons). "
        "This subtask must produce consistent numerical values and symbolic expressions to be used in subsequent calculations."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Formulate the conservation of momentum condition for the two fragments after fission, "
        "explicitly expressing the momenta of the fragments in terms of a single unknown momentum magnitude p. "
        "Emphasize that total momentum must be zero since the initial nucleus is at rest. "
        "This subtask should set up the momentum relations clearly and prepare for numerical solving of p in the next subtask."
    )
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

    cot_sc_instruction3 = (
        "Sub-task 3: Numerically solve the relativistic energy-momentum equation for the two fragments: "
        "enforce that the sum of their total energies E1 + E2 equals the initial rest energy M c^2, "
        "where each E_i = sqrt((m_i c^2)^2 + (p c)^2). Use a clear, step-by-step numerical method (e.g., Newton-Raphson or bisection) to find the momentum p. "
        "Include explicit sanity checks to verify that the resulting kinetic energies do not exceed the total available kinetic energy (1% of M c^2). "
        "This subtask must avoid the previous error of misapplying energy conservation and produce a physically consistent momentum value."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the relativistic kinetic energy T1 of the more massive fragment using the momentum p obtained from subtask_3 "
        "and the relativistic energy formula T1 = E1 - m1 c^2. Ensure numerical accuracy and clarity in the calculation. "
        "This subtask must explicitly document the calculation steps and verify that T1_rel is physically consistent (i.e., less than total kinetic energy available)."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Calculate the classical (non-relativistic) kinetic energy T1_classical of the more massive fragment using classical kinetic energy formula T = p^2/(2m) "
        "with the momentum p from subtask_3. Ensure consistency in units and numerical accuracy. "
        "This subtask should clearly contrast the classical approach with the relativistic one and prepare for difference calculation."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    cot_reflect_instruction6 = (
        "Sub-task 6: Perform a verification and sanity check by comparing the sum of relativistic kinetic energies of both fragments (using p from subtask_3) "
        "against the total available kinetic energy (1% of M c^2). Confirm that the relativistic kinetic energy T1_rel and classical kinetic energy T1_classical of the heavier fragment are physically plausible and consistent with conservation laws. "
        "This subtask aims to catch any numerical or conceptual errors before final difference calculation."
    )
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 3", "answer of subtask 3"]
    }
    results6, log6 = await self.reflexion(
        subtask_id="subtask_6",
        reflect_desc=cot_reflect_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    debate_instruction7 = (
        "Sub-task 7: Calculate the difference ΔT = T1_rel - T1_classical between the relativistic and classical kinetic energies of the more massive fragment. "
        "Interpret the numerical result in the context of the problem choices, ensuring that the difference is physically reasonable and consistent with previous verification. "
        "Provide a clear explanation linking the numerical result to the multiple-choice options."
    )
    debate_desc7 = {
        'instruction': debate_instruction7,
        'context': ["user query", results4['thinking'], results4['answer'], results5['thinking'], results5['answer'], results6['thinking'], results6['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer'], results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results7, log7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc7,
        n_repeat=self.max_round
    )
    logs.append(log7)

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'])
    return final_answer, logs
