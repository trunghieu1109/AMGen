async def forward_3(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Define and simplify the functions f(x) = ||x| - 1/2| and g(x) = ||x| - 1/4|, "
        "and express y=4g(f(sin(2πx))) and x=4g(f(cos(3πy))) in terms of x and y. "
        "Input content are results (both thinking and answer) from: []"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "final_decision_instruction": "Stage 0, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for function definitions and expressions.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1,
        n_repeat=3
    )
    logs.append(log_0_1)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(3):
        # stage_1.subtask_1
        cot_agent_instruction_1_1 = (
            f"Stage 1, Sub-task 1, Iteration {iteration+1}: Generate intermediate step representations of the graphs "
            "y=4g(f(sin(2πx))) and x=4g(f(cos(3πy))) for a given iteration, including domain partitioning and piecewise behavior. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_1.subtask_2, respectively."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_agent_instruction_1_1,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
            "temperature": 0.7,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log_1_1 = await self.answer_generate(
            subtask_id=f"stage_1.subtask_1.iter{iteration+1}",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        # stage_1.subtask_2
        cot_sc_instruction_1_2 = (
            f"Stage 1, Sub-task 2, Iteration {iteration+1}: Apply transformations and analyze intersections of the intermediate graph representations "
            "to identify candidate intersection points for the current iteration. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
        )
        final_decision_instruction_1_2 = (
            f"Stage 1, Sub-task 2, Iteration {iteration+1}, Final Decision: Synthesize and choose the most consistent answer for intersection candidates."
        )
        cot_sc_desc_1_2 = {
            "instruction": cot_sc_instruction_1_2,
            "final_decision_instruction": final_decision_instruction_1_2,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.sc_cot(
            subtask_id=f"stage_1.subtask_2.iter{iteration+1}",
            cot_agent_desc=cot_sc_desc_1_2,
            n_repeat=3
        )
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    # stage_2.subtask_1
    cot_instruction_2_1 = (
        "Stage 2, Sub-task 1: Combine intersection candidates from all loop iterations to form a consolidated set of possible intersection points. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    # stage_3.subtask_1
    aggregate_instruction_3_1 = (
        "Stage 3, Sub-task 1: Validate the consolidated intersection points against the original implicit equations and select those that satisfy both equations accurately. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    aggregate_desc_3_1 = {
        "instruction": aggregate_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.aggregate(
        subtask_id="stage_3.subtask_1",
        aggregate_desc=aggregate_desc_3_1
    )
    logs.append(log_3_1)

    # stage_4.subtask_1
    cot_reflect_instruction_4_1 = (
        "Stage 4, Sub-task 1: Count the number of valid intersection points and derive the final quantitative result. "
        "Input content are results (both thinking and answer) from: stage_3.subtask_1 & stage_0.subtask_1 & stage_1.subtask_2, respectively."
    )
    critic_instruction_4_1 = (
        "Stage 4, Sub-task 1, Criticism: Please review and provide the limitations of provided solutions of intersection counting and final result derivation."
    )
    cot_reflect_desc_4_1 = {
        "instruction": cot_reflect_instruction_4_1,
        "critic_instruction": critic_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"], results_0_1["thinking"], results_0_1["answer"], loop_results_stage_1["stage_1.subtask_2"]["thinking"], loop_results_stage_1["stage_1.subtask_2"]["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_4_1, log_4_1 = await self.reflexion(
        subtask_id="stage_4.subtask_1",
        reflect_desc=cot_reflect_desc_4_1,
        n_repeat=2
    )
    logs.append(log_4_1)

    # stage_5.subtask_1
    formatter_instruction_5_1 = (
        "Stage 5, Sub-task 1: Format the final intersection count into a clear, concise answer statement. "
        "Input content are results (both thinking and answer) from: stage_4.subtask_1, respectively."
    )
    formatter_desc_5_1 = {
        "instruction": formatter_instruction_5_1,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"],
        "format": "short and concise, without explaination"
    }
    results_5_1, log_5_1 = await self.specific_format(
        subtask_id="stage_5.subtask_1",
        formatter_desc=formatter_desc_5_1
    )
    logs.append(log_5_1)

    final_answer = await self.make_final_answer(results_5_1["thinking"], results_5_1["answer"])
    return final_answer, logs
