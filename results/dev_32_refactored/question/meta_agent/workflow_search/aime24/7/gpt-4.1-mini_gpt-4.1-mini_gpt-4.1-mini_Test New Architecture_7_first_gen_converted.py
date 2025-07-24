async def forward_7(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Rewrite the given logarithmic equations into algebraic forms involving x and y. "
        "Input content: query stating log_x(y^x) = log_y(x^{4y}) = 10, with x > 1, y > 1."
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

    # stage_1.subtask_1
    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Solve the system derived in stage_0 to find explicit values or relations for x and y. "
        "Input content: results (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    # stage_1.subtask_2
    cot_agent_instruction_1_2 = (
        "Stage 1, Sub-task 2: Calculate the product xy from the solutions obtained in stage_1.subtask_1. "
        "Input content: results (thinking and answer) from stage_1.subtask_1."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_agent_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.answer_generate(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    loop_results = {
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for i in range(3):
        aggregate_instruction_2_1 = (
            "Stage 2, Sub-task 1: Check that x and y satisfy x > 1 and y > 1 and verify the original logarithmic equations hold true. "
            "Input content: results (thinking and answer) from stage_1.subtask_2. "
            "Use all previous iteration results of this subtask for aggregation."
        )
        aggregate_desc_2_1 = {
            "instruction": aggregate_instruction_2_1,
            "input": [taskInfo] + loop_results["stage_2.subtask_1"]["answer"] + loop_results["stage_2.subtask_1"]["thinking"],
            "temperature": 0.6,
            "context": ["user query", "solutions generated from stage_2.subtask_1"]
        }
        results_2_1, log_2_1 = await self.aggregate(
            subtask_id="stage_2.subtask_1",
            aggregate_desc=aggregate_desc_2_1
        )
        logs.append(log_2_1)

        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

    # stage_3.subtask_1
    formatter_instruction_3_1 = (
        "Stage 3, Sub-task 1: Format the validated product xy into the final answer output. "
        "Input content: results from stage_1.subtask_2 and stage_2.subtask_1."
    )
    formatter_desc_3_1 = {
        "instruction": formatter_instruction_3_1,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "format": "short and concise, without explaination"
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id="stage_3.subtask_1",
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
