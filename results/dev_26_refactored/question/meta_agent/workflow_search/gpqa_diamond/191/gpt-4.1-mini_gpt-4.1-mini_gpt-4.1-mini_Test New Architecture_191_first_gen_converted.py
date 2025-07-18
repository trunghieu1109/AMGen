async def forward_191(self, taskInfo):
    logs = []

    cot_sc_instruction_stage0_sub1 = (
        "Sub-task 1: Clarify and transform the given problem parameters and geometry into a consistent vector and scalar framework, "
        "defining all relevant distances, angles, and coordinate systems for subsequent analysis."
    )
    cot_sc_desc_stage0_sub1 = {
        'instruction': cot_sc_instruction_stage0_sub1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent vector and scalar framework for the problem.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    cot_sc_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the electrostatic configuration of the uncharged spherical conductor with an off-center cavity containing charge +q, "
        "including induced charges and boundary conditions, to determine the effective source of the external electric field."
    )
    cot_sc_desc_stage1_sub1 = {
        'instruction': cot_sc_instruction_stage1_sub1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent electrostatic analysis for the problem.",
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Integrate the geometric relations (distances L, l, s and angle theta) with the electrostatic analysis "
        "to express the electric field at point P in terms of these variables, considering the conductor's shielding and induced charges."
    )
    critic_instruction_stage1_sub2 = (
        "Please review and provide the limitations of provided solutions for expressing the electric field at point P, "
        "considering the conductor's shielding and induced charges."
    )
    cot_reflect_desc_stage1_sub2 = {
        'instruction': cot_reflect_instruction_stage1_sub2,
        'critic_instruction': critic_instruction_stage1_sub2,
        'input': [
            taskInfo,
            results_stage0_sub1['thinking'], results_stage0_sub1['answer'],
            results_stage1_sub1['thinking'], results_stage1_sub1['answer']
        ],
        'temperature': 0.0,
        'context_desc': [
            "user query",
            "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"
        ]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate the candidate expressions for the electric field magnitude at point P against the derived physical model and geometric relations, "
        "to select the correct formula from the given choices."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Select the correct formula for the electric field magnitude at point P based on the physical and geometric analysis."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'final_decision_instruction': final_decision_instruction_stage2_sub1,
        'input': [
            taskInfo,
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        'context_desc': [
            "user query",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"
        ],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(
        results_stage2_sub1['thinking'],
        results_stage2_sub1['answer']
    )

    return final_answer, logs
