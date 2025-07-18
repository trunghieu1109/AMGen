async def forward_165(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the given Lagrangian, field content, and vacuum expectation values (VEVs) "
        "to understand the model setup, including the role and quantum numbers of each field. Explicitly identify the vacuum structure "
        "and the symmetry breaking pattern, ensuring clear notation of VEVs and their physical meaning. Avoid generic descriptions; focus on concrete details relevant to the model."
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

    cot_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, identify and clarify the physical nature of the pseudo-Goldstone boson H2 in the context of the model, "
        "including the mechanism of its mass generation via radiative corrections. Explicitly connect the pseudo-Goldstone boson to the approximate global symmetry and specify which couplings in the Lagrangian break this symmetry, "
        "especially highlighting the top-quark Yukawa coupling as a dominant explicit breaking source. Avoid generic pseudo-Goldstone arguments without mapping to the concrete fields and couplings."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the physical nature and mass generation mechanism of the pseudo-Goldstone boson H2, "
        "given all the above thinking and answers."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: List all one-loop contributions to the effective potential from the model’s fields (W, Z, h1, H+, H0, A0, top quark t, singlet fermions Ni), "
        "including the sign of each contribution (positive for bosons, negative for fermions) and their dependence on the VEVs. Explicitly match these loop contributions to the terms appearing in each candidate mass formula, "
        "verifying which formulae correctly include all relevant contributions and signs."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent answer for the correct loop contributions and their signs in the candidate mass formulas, "
        "given all the above thinking and answers."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2']
    }
    results3, log3 = await self.sc_cot(
        subtask_id='subtask_3',
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Perform a detailed dimensional analysis and physical consistency check of each candidate mass formula. Verify the scaling behavior of the pseudo-Goldstone boson mass squared with respect to the combined VEVs (x^2 + v^2), "
        "ensuring the formula respects expected physical limits and dimensions. Explicitly confirm whether the factor (x^2 + v^2) should appear in the numerator or denominator, and analyze the implications for the mass scale and interpretation of radiative corrections."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent answer for the dimensional and physical consistency of the candidate mass formulas, "
        "given all the above thinking and answers."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results4, log4 = await self.sc_cot(
        subtask_id='subtask_4',
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Conduct a head-to-head Debate between agents defending different candidate formulas for the pseudo-Goldstone boson mass squared. "
        "Ground the debate in explicit references to the Lagrangian couplings, loop integrals, and the results from previous subtasks. Critically evaluate the completeness, sign conventions, and scaling of each formula, "
        "forcing a rigorous comparison rather than generic consensus."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Synthesize and choose the most consistent and justified candidate formula for the pseudo-Goldstone boson mass squared, "
        "based on the debate and previous analysis."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='subtask_5',
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_reflect_instruction6 = (
        "Sub-task 6: Engage in a Reflexion phase to critically reassess the consensus reached in the Debate. Challenge assumptions about scaling, dimensionality, and physical interpretation of the candidate formulas. "
        "Ensure subtle but crucial theoretical aspects are not overlooked and that the final selection is robust both mathematically and physically. Integrate feedback to avoid overreliance on unanimous agreement without critical scrutiny."
    )
    critic_instruction6 = (
        "Please review and provide the limitations of the candidate formulas and the debate conclusions for the pseudo-Goldstone boson mass squared approximation."
    )
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'critic_instruction': critic_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4', 'thinking of subtask 5', 'answer of subtask 5']
    }
    results6, log6 = await self.reflexion(
        subtask_id='subtask_6',
        reflect_desc=cot_reflect_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    cot_instruction7 = (
        "Sub-task 7: Synthesize the comprehensive analysis from all previous subtasks to select the best approximation formula for the pseudo-Goldstone boson mass squared. "
        "Provide a detailed justification based on the model’s Lagrangian, identified symmetry-breaking sources, loop contributions, dimensional and physical consistency, and the critical evaluation from the Debate and Reflexion phases. "
        "Ensure the answer is well-founded and addresses all prior feedback and failure reasons."
    )
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer'], results5['thinking'], results5['answer'], results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking and answer of all previous subtasks']
    }
    results7, log7 = await self.sc_cot(
        subtask_id='subtask_7',
        cot_agent_desc=cot_agent_desc7,
        n_repeat=1
    )
    logs.append(log7)

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'])
    return final_answer, logs
