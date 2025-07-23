async def forward_181(self, taskInfo):
    logs = []

    cot_instruction_stage0_subtask1 = (
        "Sub-task 1: Extract and summarize the given Mott-Gurney equation, variables, parameters, and the four validity statements from the query. "
        "Input content: taskInfo containing question and choices."
    )
    cot_agent_desc_stage0_subtask1 = {
        "instruction": cot_instruction_stage0_subtask1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_stage0_subtask1, log_stage0_subtask1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_subtask1
    )
    logs.append(log_stage0_subtask1)

    loop_results_stage1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_instruction_stage1_subtask1 = (
            "Sub-task 1: Analyze the physical assumptions and relationships underlying the Mott-Gurney equation, "
            "focusing on carrier type, traps, contact type, and current components. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_1.subtask_1 and stage_1.subtask_2."
        )
        final_decision_instruction_stage1_subtask1 = (
            "Sub-task 1: Synthesize and choose the most consistent analysis of the Mott-Gurney equation assumptions."
        )
        cot_sc_desc_stage1_subtask1 = {
            "instruction": cot_sc_instruction_stage1_subtask1,
            "final_decision_instruction": final_decision_instruction_stage1_subtask1,
            "input": [
                taskInfo,
                results_stage0_subtask1["thinking"],
                results_stage0_subtask1["answer"]
            ] + loop_results_stage1["stage_1.subtask_1"]["answer"] + loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"] + loop_results_stage1["stage_1.subtask_2"]["thinking"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_stage1_subtask1, log_stage1_subtask1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_stage1_subtask1,
            n_repeat=self.max_sc
        )
        logs.append(log_stage1_subtask1)
        loop_results_stage1["stage_1.subtask_1"]["thinking"].append(results_stage1_subtask1["thinking"])
        loop_results_stage1["stage_1.subtask_1"]["answer"].append(results_stage1_subtask1["answer"])

        cot_sc_instruction_stage1_subtask2 = (
            "Sub-task 2: Classify each of the four statements against the analyzed assumptions to determine their consistency with the equation's validity. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and stage_1.subtask_1 (all iterations)."
        )
        final_decision_instruction_stage1_subtask2 = (
            "Sub-task 2: Synthesize and choose the most consistent classification of the four statements regarding the Mott-Gurney equation validity."
        )
        cot_sc_desc_stage1_subtask2 = {
            "instruction": cot_sc_instruction_stage1_subtask2,
            "final_decision_instruction": final_decision_instruction_stage1_subtask2,
            "input": [
                taskInfo,
                results_stage0_subtask1["thinking"],
                results_stage0_subtask1["answer"]
            ] + loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_stage1_subtask2, log_stage1_subtask2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_stage1_subtask2,
            n_repeat=self.max_sc
        )
        logs.append(log_stage1_subtask2)
        loop_results_stage1["stage_1.subtask_2"]["thinking"].append(results_stage1_subtask2["thinking"])
        loop_results_stage1["stage_1.subtask_2"]["answer"].append(results_stage1_subtask2["answer"])

    debate_instruction_stage2_subtask1 = (
        "Sub-task 1: Evaluate the classified statements and select the one that best satisfies the Mott-Gurney equation validity criteria. "
        "Input content: results (thinking and answer) from stage_1.subtask_1 and stage_1.subtask_2 (all iterations)."
    )
    final_decision_instruction_stage2_subtask1 = (
        "Sub-task 1: Synthesize and select the best candidate statement for the Mott-Gurney equation validity."
    )
    debate_desc_stage2_subtask1 = {
        "instruction": debate_instruction_stage2_subtask1,
        "final_decision_instruction": final_decision_instruction_stage2_subtask1,
        "input": [
            taskInfo
        ] + loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_stage2_subtask1, log_stage2_subtask1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_subtask1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_subtask1)

    cot_reflect_instruction_stage3_subtask1 = (
        "Sub-task 1: Validate the selected statement against theoretical and physical principles of the Mott-Gurney equation to confirm correctness. "
        "Input content: results (thinking and answer) from stage_2.subtask_1 and stage_1.subtask_2 (all iterations)."
    )
    critic_instruction_stage3_subtask1 = (
        "Please review and provide the limitations of the selected statement regarding the Mott-Gurney equation validity."
    )
    cot_reflect_desc_stage3_subtask1 = {
        "instruction": cot_reflect_instruction_stage3_subtask1,
        "critic_instruction": critic_instruction_stage3_subtask1,
        "input": [
            taskInfo,
            results_stage2_subtask1["thinking"],
            results_stage2_subtask1["answer"]
        ] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_stage3_subtask1, log_stage3_subtask1 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_stage3_subtask1,
        n_repeat=self.max_round
    )
    logs.append(log_stage3_subtask1)

    final_answer = await self.make_final_answer(results_stage3_subtask1["thinking"], results_stage3_subtask1["answer"])
    return final_answer, logs
