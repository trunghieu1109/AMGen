async def forward_165(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the defining features of the given BSM Lagrangian, including particle content, "
        "quantum numbers, vacuum expectation values, and relevant interactions. Ensure clarity on the scalar and fermion sectors and their roles in symmetry breaking and mass generation. "
        "This subtask sets the foundation for understanding the physical context and must avoid superficial summaries that omit key details."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage1_subtask1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Identify and explicitly list all relevant one-loop radiative correction contributions to the pseudo-Goldstone boson mass, "
        "including bosonic (h1, W, Z, H±, H0, A0) and fermionic (top quark t, singlet fermions N_i) loops. For each contribution, specify the expected sign (positive for bosons, negative for fermions), "
        "the power of the mass term (quartic), and the role of coefficients alpha_i. This subtask must embed the feedback to avoid omission of dominant contributions such as the top quark and ensure no assumptions without explicit verification."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage1_subtask2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Perform a detailed term-by-term comparison of the identified loop contributions against each candidate formula. "
        "Verify the presence, sign, and coefficient structure of each particle's contribution, with special attention to the top quark term and the singlet fermion terms. "
        "Additionally, analyze the placement and dimensional consistency of the vacuum expectation value factor (x^2 + v^2) in numerator or denominator. "
        "This subtask must explicitly address the previous failure of ignoring the top quark term and the incorrect dimensional scaling, ensuring a physically consistent and complete mapping."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage1_subtask2", "answer of stage1_subtask2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage1_subtask3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Conduct an intermediate verification and reflexion step to self-audit the mapping of physical contributions to candidate formulas. "
        "This includes cross-checking for any missing terms, sign inconsistencies, or dimensional mismatches. Agents must flag any discrepancies or ambiguities before proceeding to final selection. "
        "This step is critical to prevent confirmation bias and to ensure that the final choice is robust and physically justified."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage1_subtask3", "answer of stage1_subtask3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="stage1_subtask4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Stage 2 Sub-task 1: Derive the approximate mass formula for the pseudo-Goldstone boson H_2 based on the processed inputs and verified loop contributions. "
        "Then, select the correct candidate formula by matching the derived expression with the candidates, ensuring all dominant contributions (including the top quark) are included with correct signs and that the formula is dimensionally consistent. "
        "Finally, provide a clear justification referencing the verification step to avoid previous errors of premature or biased selection."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage1_subtask4", "answer of stage1_subtask4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="stage2_subtask1",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])

    return final_answer, logs
