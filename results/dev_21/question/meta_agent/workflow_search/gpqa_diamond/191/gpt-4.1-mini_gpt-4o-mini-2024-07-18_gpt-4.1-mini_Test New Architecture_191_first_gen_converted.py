async def forward_191(self, taskInfo):
    logs = []

    cot_sc_instruction0 = (
        "Sub-task 1: Analyze and classify the geometric and physical elements of the problem, "
        "including the spherical conductor, cavity, charge placement, and point P, and clarify vector relationships and parameters (R, r, s, q, L, l, theta)."
    )
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    debate_instruction1 = (
        "Sub-task 1: Generate and evaluate possible electrostatic configurations and field expressions at point P, "
        "considering the conductor's shielding effect, induced charges, and the displacement of the cavity (off-center), "
        "to prioritize plausible formulas for the electric field magnitude."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'context': ["user query", results0['thinking'], results0['answer']],
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 1: Compute or identify the correct quantitative expression for the magnitude of the electric field at point P outside the conductor, "
        "using the evaluated configurations and geometric relations, and verify which given choice matches the derived formula."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer'], results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", results0['thinking'], results0['answer'], results1['thinking'], results1['answer']]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    final_answer = await self.make_final_answer(results2['thinking'], results2['answer'])
    return final_answer, logs
