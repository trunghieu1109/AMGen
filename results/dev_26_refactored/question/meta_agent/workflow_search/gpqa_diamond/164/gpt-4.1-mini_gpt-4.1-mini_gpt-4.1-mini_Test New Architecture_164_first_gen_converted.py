async def forward_164(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Extract and transform the given information about the polymerization setup, catalyst systems, "
        "and the four statements into a structured format suitable for analysis."
    )
    cot_agent_desc_stage0 = {
        "instruction": cot_instruction_stage0,
        "input": [taskInfo],
        "temperature": 0.5,
        "context": ["user query"]
    }
    results_stage0, log_stage0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0)

    cot_instruction_stage1 = (
        "Sub-task 1: Combine and integrate the extracted information with relevant chemical knowledge about ethylene polymerization, "
        "catalyst types (group VIa metals, noble metals), activators, and industrial implementation to understand the feasibility and constraints of the dual catalyst system."
    )
    final_decision_instruction_stage1 = (
        "Sub-task 1: Synthesize and choose the most consistent understanding of the dual catalyst system feasibility and constraints."
    )
    cot_agent_desc_stage1 = {
        "instruction": cot_instruction_stage1,
        "final_decision_instruction": final_decision_instruction_stage1,
        "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1, log_stage1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_stage1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Evaluate the four statements given by the senior scientist regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system. "
        "Use the integrated chemical and industrial knowledge to select the correct statement."
    )
    final_decision_instruction_stage2 = (
        "Sub-task 1: Select the correct statement regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system."
    )
    debate_desc_stage2 = {
        "instruction": debate_instruction_stage2,
        "final_decision_instruction": final_decision_instruction_stage2,
        "input": [taskInfo, results_stage1["thinking"], results_stage1["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage2["thinking"], results_stage2["answer"])

    return final_answer, logs
