async def forward_179(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and explicitly list all given numerical data, constants, and problem constraints from the query, "
        "including number of charges (N=13), charge magnitude (q=2e), radius (R=2 m), and the fixed position of the 13th charge at point P. "
        "Avoid assumptions and cross-verify with the original problem statement."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc={
            'instruction': cot_instruction1,
            'final_decision_instruction': "Sub-task 1: Provide a clear, explicit list of all numerical data and constraints extracted from the query.",
            'input': [taskInfo],
            'context_desc': ["user query"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Formulate the mathematical expressions for the electrostatic potential energy of the system using the extracted parameters: "
        "write explicit sum formulas for (a) the interaction energy between the central charge and each of the 12 charges on the sphere, "
        "and (b) the interaction energy among the 12 charges themselves constrained on the sphere. Incorporate known results from the Thomson problem for the minimum energy configuration of 12 identical charges on a unit sphere, scaled appropriately."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "output of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc={
            'instruction': cot_instruction2,
            'final_decision_instruction': "Sub-task 2: Provide explicit mathematical expressions for the electrostatic potential energy components as described.",
            'input': [taskInfo, results1['answer']],
            'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Compute numerically the electrostatic potential energy contribution from the interaction between the central charge at point P and the 12 charges on the sphere, "
        "using Coulomb's law with q=2e and R=2 m. Ensure units are consistent and calculations are explicit."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent numeric value for the central-to-surface interaction energy, "
        "given the calculations and context."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_2.subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Compute numerically the electrostatic potential energy contribution from the interactions among the 12 charges on the sphere, "
        "using the known minimum energy value from the Thomson problem for 12 unit charges on a unit sphere, scaled by q=2e and R=2 m. "
        "Explicitly perform the scaling and unit conversion to Joules."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent numeric value for the surface-to-surface interaction energy, "
        "given the calculations and context."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_2.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Sum the computed energy contributions from the central-to-surface and surface-to-surface interactions to obtain the total minimum electrostatic potential energy of the system. "
        "Verify the result is in Joules and round to three decimal places as requested."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of the provided total energy calculation and verify unit consistency and rounding correctness."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="stage_3.subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_sc_instruction6 = (
        "Sub-task 6: Compare the computed total minimum energy with the provided answer choices, select the correct one, "
        "and justify the choice based on numeric closeness and physical plausibility."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Provide the final answer choice and justification for the minimum energy of the system."
    )
    cot_sc_desc6 = {
        'instruction': cot_sc_instruction6,
        'final_decision_instruction': final_decision_instruction6,
        'input': [taskInfo, results5['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "answer of subtask 5"]
    }
    results6, log6 = await self.sc_cot(
        subtask_id="stage_3.subtask_6",
        cot_agent_desc=cot_sc_desc6,
        n_repeat=self.max_sc
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
