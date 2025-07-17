async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and define all relevant physical quantities and parameters from the problem statement, "
        "including initial nucleus mass M (rest-mass energy 300 GeV), rest masses of two fragments m and 2m, total rest mass after fission (0.99 M), "
        "and mass ratio. Provide precise numerical values or expressions for all parameters to be used in subsequent calculations. "
        "Avoid skipping or assuming values."
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

    cot_instruction2 = (
        "Sub-task 2: Apply conservation of momentum and total energy principles to derive the relationship between the fragments' momenta and energies. "
        "Explicitly set up the relativistic energy-momentum conservation equation: sqrt(m1^2 c^4 + p^2 c^2) + sqrt(m2^2 c^4 + p^2 c^2) = M c^2, "
        "where m1 and m2 are fragment rest masses. Solve for the common momentum p numerically. "
        "Prepare a solvable equation for p for kinetic energy calculations."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Numerically solve for the common fragment momentum p from the relativistic energy conservation equation derived in subtask_2. "
        "Use explicit numerical methods or approximations to find p with sufficient precision. "
        "Output p for accurate relativistic kinetic energy calculations."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the relativistic kinetic energy T1 of the more massive fragment using the solved momentum p from subtask_3. "
        "Compute Lorentz factor gamma = sqrt(1 + (p/(m1 c))^2), then T1_rel = (gamma - 1) * m1 * c^2 numerically. "
        "Show all numerical steps and intermediate values explicitly."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Calculate the classical (non-relativistic) kinetic energy T1_classical of the more massive fragment using the same momentum p from subtask_3. "
        "Use T1_classical = p^2 / (2 m1) and compute numerically with explicit arithmetic."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Determine the difference Delta T = T1_rel - T1_classical between relativistic and classical kinetic energies of the more massive fragment. "
        "Perform explicit numerical subtraction and interpret the result in context of problem choices. "
        "Cross-validate magnitude of relativistic corrections for consistency."
    )
    debate_desc6 = {
        'instruction': debate_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
