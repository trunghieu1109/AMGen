async def forward_168(self, taskInfo):
    logs = []

    debate_instruction_stage0 = (
        "Sub-task 1: Analyze and classify the given decay process elements, including particle types, masses, "
        "and the original energy spectrum characteristics, based on the query and given information."
    )
    debate_final_decision_stage0 = (
        "Sub-task 1: Provide a clear classification and summary of the decay elements and spectrum characteristics."
    )
    debate_desc_stage0 = {
        "instruction": debate_instruction_stage0,
        "final_decision_instruction": debate_final_decision_stage0,
        "input": [taskInfo],
        "context_desc": ["user query"],
        "temperature": 0.5
    }
    results_stage0, log_stage0 = await self.debate(
        subtask_id="stage_0.subtask_1",
        debate_desc=debate_desc_stage0,
        n_repeat=self.max_round
    )
    logs.append(log_stage0)

    debate_instruction_stage1 = (
        "Sub-task 1: Assess the impact of replacing two V particles with one massless M particle on the kinematics "
        "and energy distribution of the decay products, using the output from stage_0.subtask_1."
    )
    debate_final_decision_stage1 = (
        "Sub-task 1: Provide a detailed assessment of how the variant decay affects the energy distribution and kinematics."
    )
    debate_desc_stage1 = {
        "instruction": debate_instruction_stage1,
        "final_decision_instruction": debate_final_decision_stage1,
        "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"]],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    cot_sc_instruction_stage2 = (
        "Sub-task 1: Based on the output from stage_1.subtask_1, derive the expected changes in the total energy spectrum "
        "of the outgoing E particles, focusing on continuity, shape, and endpoint modifications due to the variant decay."
    )
    final_decision_instruction_stage2 = (
        "Sub-task 1: Synthesize and choose the most consistent and correct conclusions about the energy spectrum changes."
    )
    cot_sc_desc_stage2 = {
        "instruction": cot_sc_instruction_stage2,
        "final_decision_instruction": final_decision_instruction_stage2,
        "input": [taskInfo, results_stage1["thinking"], results_stage1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage2, log_stage2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc_stage2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage2)

    cot_sc_instruction_stage3 = (
        "Sub-task 1: Combine and transform quantitative inputs such as particle masses, number of emitted particles, "
        "and conservation laws to quantify how the endpoint and shape of the E particle energy spectrum adjust in the variant decay, "
        "using outputs from stage_0.subtask_1 and stage_2.subtask_1."
    )
    final_decision_instruction_stage3 = (
        "Sub-task 1: Provide a consistent quantitative summary of the endpoint and shape changes in the energy spectrum."
    )
    cot_sc_desc_stage3 = {
        "instruction": cot_sc_instruction_stage3,
        "final_decision_instruction": final_decision_instruction_stage3,
        "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"], results_stage2["thinking"], results_stage2["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_stage3, log_stage3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc_stage3,
        n_repeat=self.max_sc
    )
    logs.append(log_stage3)

    final_answer = await self.make_final_answer(results_stage3["thinking"], results_stage3["answer"])

    return final_answer, logs
