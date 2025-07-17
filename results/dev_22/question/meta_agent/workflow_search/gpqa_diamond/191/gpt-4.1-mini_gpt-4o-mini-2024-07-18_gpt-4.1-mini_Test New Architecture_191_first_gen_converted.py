async def forward_191(self, taskInfo):
    logs = []

    cot_sc_instruction0_1 = "Sub-task 1: Analyze and classify the physical elements and parameters given in the problem: the spherical conductor, cavity, charge placement, and points of interest (P), including their geometric and electrostatic relationships."
    cot_sc_desc0_1 = {
        'instruction': cot_sc_instruction0_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    debate_instruction1_1 = "Sub-task 1: Generate and evaluate possible electrostatic configurations and interpretations of the induced charges and fields due to the displaced cavity and internal charge, considering conductor properties and boundary conditions."
    debate_desc1_1 = {
        'instruction': debate_instruction1_1,
        'context': ["user query", results0_1['thinking'], results0_1['answer']],
        'input': [taskInfo, results0_1['thinking'], results0_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_sc_instruction1_2 = "Sub-task 2: Prioritize and select the most physically consistent and mathematically plausible expressions for the electric field at point P based on geometric relations and conductor shielding effects."
    cot_sc_desc1_2 = {
        'instruction': cot_sc_instruction1_2,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    cot_sc_instruction2_1 = "Sub-task 1: Compute or identify the magnitude of the electric field at point P outside the spherical conductor using the selected configuration and expressions, verifying the dependence on distances L, l, s, and angle theta."
    cot_sc_desc2_1 = {
        'instruction': cot_sc_instruction2_1,
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'temperature': 0.0,
        'context': ["user query", results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])
    return final_answer, logs
