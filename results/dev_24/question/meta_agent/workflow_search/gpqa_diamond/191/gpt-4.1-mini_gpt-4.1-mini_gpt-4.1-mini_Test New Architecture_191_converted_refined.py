async def forward_191(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and clearly define the physical setup of the spherical conductor, cavity, charge placement, "
        "and observation point. Explicitly clarify vector definitions and geometric relationships of distances l, L, s, and angle theta, "
        "preventing confusion between l and L, with context from the user query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results_stage1_sub1, log1 = await self.debate(
        subtask_id="stage1_subtask1",
        debate_desc={
            'instruction': cot_instruction1,
            'final_decision_instruction': "Sub-task 1: Provide a clear and unambiguous description of the physical setup and vector definitions.",
            'input': [taskInfo],
            'context_desc': ["user query"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Examine electrostatic principles governing the system, focusing on the conductor's shielding effect and uniqueness theorem. "
        "Justify that the external electric field depends solely on the total induced charge on the conductor's outer surface, "
        "behaving as a point charge +q at the conductor's center. Refine understanding iteratively with context from previous subtask."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'critic_instruction': "Please review and provide limitations and refinements of the electrostatic shielding explanation.",
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results_stage1_sub2, log2 = await self.reflexion(
        subtask_id="stage1_subtask2",
        reflect_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 1: Derive the expression for the magnitude of the electric field at point P outside the conductor, "
        "based on the shielding principle established in stage1_subtask2. Ensure the formula depends on the correct distance variable L, "
        "not on l or s. Incorporate physical insights from previous subtasks to avoid incorrect distance usage."
    )
    final_decision_instruction3 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct formula for the electric field magnitude at point P, "
        "given all previous thinking and answers."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1", "thinking of stage1_subtask2", "answer of stage1_subtask2"]
    }
    results_stage2_sub1, log3 = await self.sc_cot(
        subtask_id="stage2_subtask1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 2: Perform explicit verification to confirm which distance variable (l or L) appears in Coulomb's law for the external field of a point charge located at the conductor's center. "
        "Include a numeric sanity check on how the field scales as point P moves farther from the conductor center, reinforcing correct variable assignment."
    )
    final_decision_instruction4 = (
        "Sub-task 2: Provide a final decision on the correct distance variable for the electric field formula, supported by numeric sanity check."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'context_desc': ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1"],
        'temperature': 0.5
    }
    results_stage2_sub2, log4 = await self.debate(
        subtask_id="stage2_subtask2",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 1: Integrate all prior analyses and verifications to select the correct formula for the magnitude of the electric field at point P from the given choices. "
        "Explicitly reference the shielding principle, uniqueness theorem, and verified correct distance variable L. Ensure consistency with fundamental electrostatics."
    )
    final_decision_instruction5 = (
        "Sub-task 1: Provide the final answer selecting the correct formula for the electric field magnitude at point P, with justification."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results_stage2_sub1['thinking'], results_stage2_sub1['answer'], results_stage2_sub2['thinking'], results_stage2_sub2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage2_subtask1", "answer of stage2_subtask1", "thinking of stage2_subtask2", "answer of stage2_subtask2"]
    }
    results_stage3_sub1, log5 = await self.sc_cot(
        subtask_id="stage3_subtask1",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results_stage3_sub1['thinking'], results_stage3_sub1['answer'])
    return final_answer, logs
