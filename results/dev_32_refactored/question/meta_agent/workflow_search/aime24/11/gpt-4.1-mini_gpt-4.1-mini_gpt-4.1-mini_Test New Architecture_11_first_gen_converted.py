async def forward_11(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_sc_instruction = (
        "Stage 0, Sub-task 1: Define the problem parameters and represent paths as sequences of moves with exactly 4 direction changes on the 8x8 grid. "
        "Input content: [taskInfo]"
    )
    final_decision_instruction = (
        "Stage 0, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for problem parameter definition and path representation."
    )
    cot_sc_desc = {
        "instruction": cot_sc_instruction,
        "final_decision_instruction": final_decision_instruction,
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results_stage0, log0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc,
        n_repeat=3
    )
    logs.append(log0)

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(3):
        cot_instruction_1 = (
            "Stage 1, Sub-task 1: Combine intermediate path segments by applying direction change counting rules to produce consolidated partial counts. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_2.subtask_1, respectively."
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo, results_stage0['thinking'], results_stage0['answer']] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"] + ["thinking of previous stage_2.subtask_1"]*len(loop_results["stage_2.subtask_1"]["thinking"]) + ["answer of previous stage_2.subtask_1"]*len(loop_results["stage_2.subtask_1"]["answer"])
        }
        results_stage1, log1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1
        )
        logs.append(log1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_stage1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_stage1["answer"])

        aggregate_instruction_2 = (
            "Stage 2, Sub-task 1: Validate partial path counts to ensure exactly 4 direction changes and assess their correctness for aggregation. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
        )
        aggregate_desc_2 = {
            "instruction": aggregate_instruction_2,
            "input": [taskInfo, results_stage1["thinking"], results_stage1["answer"]],
            "temperature": 0.7,
            "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_stage2, log2 = await self.aggregate(
            subtask_id="stage_2.subtask_1",
            aggregate_desc=aggregate_desc_2
        )
        logs.append(log2)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_stage2["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_stage2["answer"])

    cot_reflect_instruction_3 = (
        "Stage 3, Sub-task 1: Derive the total number of valid paths by consolidating validated partial counts and applying combinatorial formulas. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_2.subtask_1, respectively."
    )
    critic_instruction_3 = (
        "Stage 3, Sub-task 1, Criticism: Please review and provide the limitations of provided solutions of total path count calculation."
    )
    cot_reflect_desc_3 = {
        "instruction": cot_reflect_instruction_3,
        "critic_instruction": critic_instruction_3,
        "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"]] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"] + ["thinking of stage_2.subtask_1"]*len(loop_results["stage_2.subtask_1"]["thinking"]) + ["answer of stage_2.subtask_1"]*len(loop_results["stage_2.subtask_1"]["answer"])
    }
    results_stage3, log3 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_3,
        n_repeat=1
    )
    logs.append(log3)

    formatter_instruction_4 = (
        "Stage 4, Sub-task 1: Format the computed number of paths into a clear, concise final answer. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_3.subtask_1, respectively."
    )
    formatter_desc_4 = {
        "instruction": formatter_instruction_4,
        "input": [taskInfo, loop_results["stage_1.subtask_1"]["answer"], results_stage3["thinking"], results_stage3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "answers of stage_1.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "format": "short and concise, without explaination"
    }
    results_stage4, log4 = await self.specific_format(
        subtask_id="stage_4.subtask_1",
        formatter_desc=formatter_desc_4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results_stage4["thinking"], results_stage4["answer"])
    return final_answer, logs
