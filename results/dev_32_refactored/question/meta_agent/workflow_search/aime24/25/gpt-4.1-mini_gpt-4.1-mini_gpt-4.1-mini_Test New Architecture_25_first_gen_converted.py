async def forward_25(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Analyze the properties of the convex equilateral hexagon with parallel opposite sides and characterize the triangle formed by extensions of sides AB, CD, and EF, including the relationship between the hexagon side length and the triangle side lengths 200, 240, and 300. Input content are results (both thinking and answer) from: none."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(3):
        cot_sc_instruction_1_1 = (
            "Stage 1, Sub-task 1: Construct initial intermediate expressions relating the hexagon side length to the triangle side lengths using vector or coordinate geometry based on the analysis from stage_0.subtask_1. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        final_decision_instruction_1_1 = (
            "Stage 1, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for constructing intermediate expressions relating the hexagon side length to the triangle side lengths."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=3
        )
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])
        logs.append(log_1_1)

        revise_instruction_1_2 = (
            "Stage 1, Sub-task 2: Refine and simplify the intermediate expressions to isolate the hexagon side length, incorporating feedback from previous iteration results if applicable. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
        )
        revise_desc_1_2 = {
            "instruction": revise_instruction_1_2,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.revise(
            subtask_id="stage_1.subtask_2",
            revise_desc=revise_desc_1_2
        )
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])
        logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Stage 2, Sub-task 1: Evaluate candidate hexagon side lengths derived from the refined expressions and select the value that best satisfies the geometric constraints and given triangle side lengths. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    final_decision_instruction_2_1 = (
        "Stage 2, Sub-task 1, Final Decision: Select the best candidate hexagon side length satisfying the problem conditions."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=2
    )
    logs.append(log_2_1)

    cot_instruction_3_1 = (
        "Stage 3, Sub-task 1: Validate the selected hexagon side length against the problem conditions for correctness, consistency, and geometric feasibility, producing a final assessment. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2 & stage_2.subtask_1, respectively."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + [results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
