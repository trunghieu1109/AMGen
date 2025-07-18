async def forward_164(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction_stage0 = (
        "Sub-task 1: Extract and transform the given information from the query into a structured summary highlighting key entities, constraints, and statements for downstream analysis."
    )
    cot_sc_desc_stage0 = {
        "instruction": cot_sc_instruction_stage0,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent structured summary of the given information.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results_stage0, log_stage0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc_stage0,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0)

    cot_sc_instruction_stage1_subtask1 = (
        "Sub-task 1: Combine and integrate chemical knowledge about ethylene polymerization, dual catalyst systems, group VIa transition metals, activators (especially aluminum-based), and noble metal catalysts with the extracted information to understand feasibility and mechanisms."
    )
    cot_sc_desc_stage1_subtask1 = {
        "instruction": cot_sc_instruction_stage1_subtask1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent integrated chemical knowledge.",
        "input": [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_subtask1, log_stage1_subtask1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_stage1_subtask1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_subtask1)

    cot_reflect_instruction_stage1_subtask2 = (
        "Sub-task 2: Incorporate industrial and economic considerations regarding the implementation of dual catalyst systems in the US and the cost implications of noble metal catalysts."
    )
    critic_instruction_stage1_subtask2 = (
        "Please review and provide the limitations of provided solutions regarding industrial implementation and economic considerations of dual catalyst systems and noble metal catalysts."
    )
    cot_reflect_desc_stage1_subtask2 = {
        "instruction": cot_reflect_instruction_stage1_subtask2,
        "critic_instruction": critic_instruction_stage1_subtask2,
        "input": [taskInfo, results_stage0['thinking'], results_stage0['answer'], results_stage1_subtask1['thinking'], results_stage1_subtask1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage1_subtask2, log_stage1_subtask2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_stage1_subtask2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_subtask2)

    debate_instruction_stage2 = (
        "Sub-task 1: Evaluate each of the four statements against the integrated chemical, industrial, and economic knowledge to identify which statement is correct regarding the formation of branched polyethylene using only ethylene and a dual catalyst system."
    )
    final_decision_instruction_stage2 = (
        "Sub-task 1: Determine the correct statement among the four provided regarding the dual catalyst system for branched polyethylene formation from ethylene."
    )
    debate_desc_stage2 = {
        "instruction": debate_instruction_stage2,
        "final_decision_instruction": final_decision_instruction_stage2,
        "input": [taskInfo, results_stage1_subtask1['thinking'], results_stage1_subtask1['answer'], results_stage1_subtask2['thinking'], results_stage1_subtask2['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_stage2_subtask1, log_stage2_subtask1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_subtask1)

    final_answer = await self.make_final_answer(results_stage2_subtask1['thinking'], results_stage2_subtask1['answer'])
    return final_answer, logs
