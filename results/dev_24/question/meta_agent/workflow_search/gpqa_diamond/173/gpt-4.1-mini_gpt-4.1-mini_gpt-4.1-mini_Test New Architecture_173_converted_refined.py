async def forward_173(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize all given quantitative data and physical constraints from the problem statement, "
        "including initial nucleus rest-mass energy, fragment mass ratio, total rest mass after fission, and conservation laws. "
        "Ensure clarity on assumptions such as ignoring electrons and no other particles emitted. "
        "This subtask sets the foundation for all subsequent numeric calculations."
    )
    cot_sc_final_decision1 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct summary of given data and physical constraints "
        "from the problem statement."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': cot_sc_final_decision1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Derive explicit numerical values for the rest masses of the two fragments based on the given mass ratio (2:1) "
        "and total rest mass after fission (99% of initial mass). Calculate the total kinetic energy available from the 1% mass loss (converted to energy). "
        "This subtask must produce concrete numeric values (in GeV) for fragment masses and total kinetic energy to be used downstream."
    )
    debate_final_decision2 = (
        "Sub-task 2: Synthesize and select the most consistent and correct numeric values for fragment masses and total kinetic energy "
        "based on the problem data and previous summary."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': debate_final_decision2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_reflect_instruction3a = (
        "Sub-task 3a: Numerically solve the relativistic energy conservation equation for the fragment momentum p, "
        "given the initial rest-mass energy (300 GeV) and fragment rest masses from previous subtask. "
        "This step must produce an explicit numeric value for p (in GeV/c)."
    )
    critic_instruction3a = (
        "Please review and provide limitations or improvements for the numeric solution of fragment momentum p."
    )
    cot_reflect_desc3a = {
        'instruction': cot_reflect_instruction3a,
        'critic_instruction': critic_instruction3a,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3a, log3a = await self.reflexion(
        subtask_id="subtask_3a",
        reflect_desc=cot_reflect_desc3a,
        n_repeat=self.max_round
    )
    logs.append(log3a)

    cot_reflect_instruction3b = (
        "Sub-task 3b: Using the momentum p from subtask 3a, compute the velocity v1 and Lorentz factor gamma1 of the heavier fragment explicitly "
        "via relativistic formulas. Provide numeric values for v1 (as fraction of c) and gamma1."
    )
    critic_instruction3b = (
        "Please review and provide limitations or improvements for the velocity and Lorentz factor calculations."
    )
    cot_reflect_desc3b = {
        'instruction': cot_reflect_instruction3b,
        'critic_instruction': critic_instruction3b,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3a['thinking'], results3a['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3a", "answer of subtask 3a"]
    }
    results3b, log3b = await self.reflexion(
        subtask_id="subtask_3b",
        reflect_desc=cot_reflect_desc3b,
        n_repeat=self.max_round
    )
    logs.append(log3b)

    cot_instruction4a = (
        "Sub-task 4a: Calculate the relativistic kinetic energy T1_rel of the heavier fragment using T1_rel = (gamma1 - 1) * m1 * c^2 "
        "with numeric values from subtask 3b. Provide explicit numeric calculation in GeV or MeV."
    )
    cot_agent_desc4a = {
        'instruction': cot_instruction4a,
        'input': [taskInfo, results3b['thinking'], results3b['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4a, log4a = await self.cot(
        subtask_id="subtask_4a",
        cot_agent_desc=cot_agent_desc4a
    )
    logs.append(log4a)

    cot_instruction4b = (
        "Sub-task 4b: Calculate the classical (non-relativistic) kinetic energy T1_cl of the heavier fragment using T1_cl = p^2 / (2 * m1) "
        "with numeric momentum p from subtask 3a and mass m1 from subtask 2. Provide explicit numeric evaluation in GeV or MeV."
    )
    cot_agent_desc4b = {
        'instruction': cot_instruction4b,
        'input': [taskInfo, results3a['thinking'], results3a['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4b, log4b = await self.cot(
        subtask_id="subtask_4b",
        cot_agent_desc=cot_agent_desc4b
    )
    logs.append(log4b)

    debate_instruction5 = (
        "Sub-task 5: Compute the numeric difference Delta_T = T1_rel - T1_cl explicitly in MeV. "
        "Compare this difference to the multiple-choice options and select the correct answer. "
        "Include a brief justification referencing the numeric values to support the final choice."
    )
    debate_final_decision5 = (
        "Sub-task 5: Synthesize and select the correct multiple-choice answer based on the numeric difference computed."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': debate_final_decision5,
        'input': [taskInfo, results4a['thinking'], results4a['answer'], results4b['thinking'], results4b['answer']],
        'context_desc': ["user query", "thinking of subtask 4a", "answer of subtask 4a", "thinking of subtask 4b", "answer of subtask 4b"],
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
