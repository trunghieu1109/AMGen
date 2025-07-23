async def forward_161(self, taskInfo):
    logs = []
    results = {}

    cot_instruction_0_1 = (
        "Subtask stage_0.subtask_1: Identify and extract the given metric, variables, domain, radius, and answer choices from the query. "
        "Input: taskInfo containing the question and choices."
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
    results["stage_0.subtask_1"] = results_0_1

    cot_instruction_1_1 = (
        "Subtask stage_1.subtask_1: Analyze the metric's geometric meaning, domain constraints, and interpret the radius in context of the pseudosphere. "
        "Input: results from stage_0.subtask_1 (thinking and answer)."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)
    results["stage_1.subtask_1"] = results_1_1

    loop_results_stage_2 = {
        "stage_2.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_2_1 = (
            f"Iteration {iteration+1} - Subtask stage_2.subtask_1: Formulate the area integral expression for the pseudosphere using the given metric and radius. "
            "Input: results from stage_1.subtask_1 (thinking and answer) and all previous iteration outputs of stage_2 subtasks."
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]] + \
                     loop_results_stage_2["stage_2.subtask_1"]["answer"] + loop_results_stage_2["stage_2.subtask_1"]["thinking"] + \
                     loop_results_stage_2["stage_2.subtask_2"]["answer"] + loop_results_stage_2["stage_2.subtask_2"]["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"] + \
                            ["answer of previous iterations stage_2.subtask_1"] * len(loop_results_stage_2["stage_2.subtask_1"]["answer"]) + \
                            ["thinking of previous iterations stage_2.subtask_1"] * len(loop_results_stage_2["stage_2.subtask_1"]["thinking"]) + \
                            ["answer of previous iterations stage_2.subtask_2"] * len(loop_results_stage_2["stage_2.subtask_2"]["answer"]) + \
                            ["thinking of previous iterations stage_2.subtask_2"] * len(loop_results_stage_2["stage_2.subtask_2"]["thinking"])
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results_stage_2["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results_stage_2["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

        cot_instruction_2_2 = (
            f"Iteration {iteration+1} - Subtask stage_2.subtask_2: Evaluate or simplify the area integral, considering boundary behavior and singularities. "
            "Input: results from stage_2.subtask_1 (thinking and answer) of all iterations."
        )
        cot_agent_desc_2_2 = {
            "instruction": cot_instruction_2_2,
            "input": [taskInfo] + loop_results_stage_2["stage_2.subtask_1"]["answer"] + loop_results_stage_2["stage_2.subtask_1"]["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query"] + \
                            ["answer of all iterations stage_2.subtask_1"] * len(loop_results_stage_2["stage_2.subtask_1"]["answer"]) + \
                            ["thinking of all iterations stage_2.subtask_1"] * len(loop_results_stage_2["stage_2.subtask_1"]["thinking"])
        }
        results_2_2, log_2_2 = await self.cot(
            subtask_id="stage_2.subtask_2",
            cot_agent_desc=cot_agent_desc_2_2
        )
        logs.append(log_2_2)
        loop_results_stage_2["stage_2.subtask_2"]["thinking"].append(results_2_2["thinking"])
        loop_results_stage_2["stage_2.subtask_2"]["answer"].append(results_2_2["answer"])

    results["stage_2.subtask_1"] = {
        "thinking": loop_results_stage_2["stage_2.subtask_1"]["thinking"],
        "answer": loop_results_stage_2["stage_2.subtask_1"]["answer"]
    }
    results["stage_2.subtask_2"] = {
        "thinking": loop_results_stage_2["stage_2.subtask_2"]["thinking"],
        "answer": loop_results_stage_2["stage_2.subtask_2"]["answer"]
    }

    cot_instruction_3_1 = (
        "Subtask stage_3.subtask_1: Combine intermediate results to determine the final area value and compare it with the provided answer choices. "
        "Input: results from stage_2.subtask_2 (thinking and answer) of all iterations."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo] + results["stage_2.subtask_2"]["answer"] + results["stage_2.subtask_2"]["thinking"],
        "temperature": 0.0,
        "context_desc": ["user query"] + \
                        ["answer of all iterations stage_2.subtask_2"] * len(results["stage_2.subtask_2"]["answer"]) + \
                        ["thinking of all iterations stage_2.subtask_2"] * len(results["stage_2.subtask_2"]["thinking"])
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)
    results["stage_3.subtask_1"] = results_3_1

    aggregate_instruction_4_1 = (
        "Subtask stage_4.subtask_1: Select the correct answer choice based on the consolidated area computation and reasoning. "
        "Input: results from stage_3.subtask_1 (thinking and answer)."
    )
    aggregate_desc_4_1 = {
        "instruction": aggregate_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_4_1, log_4_1 = await self.aggregate(
        subtask_id="stage_4.subtask_1",
        aggregate_desc=aggregate_desc_4_1
    )
    logs.append(log_4_1)
    results["stage_4.subtask_1"] = results_4_1

    cot_reflect_instruction_5_1 = (
        "Subtask stage_5.subtask_1: Validate the selected answer for correctness and consistency with the problem statement and mathematical principles. "
        "Input: results from stage_4.subtask_1 (thinking and answer)."
    )
    critic_instruction_5_1 = (
        "Please review and provide the limitations of the provided solution for the selected answer choice, checking for correctness and consistency."
    )
    cot_reflect_desc_5_1 = {
        "instruction": cot_reflect_instruction_5_1,
        "critic_instruction": critic_instruction_5_1,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"]
    }
    results_5_1, log_5_1 = await self.reflexion(
        subtask_id="stage_5.subtask_1",
        reflect_desc=cot_reflect_desc_5_1,
        n_repeat=1
    )
    logs.append(log_5_1)
    results["stage_5.subtask_1"] = results_5_1

    formatter_instruction_6_1 = (
        "Subtask stage_6.subtask_1: Format the final answer in LaTeX and provide a concise summary of the solution. "
        "Input: results from stage_5.subtask_1 (thinking and answer)."
    )
    formatter_desc_6_1 = {
        "instruction": formatter_instruction_6_1,
        "input": [taskInfo, results_5_1["thinking"], results_5_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_5.subtask_1", "answer of stage_5.subtask_1"],
        "format": "short and concise, without explanation"
    }
    results_6_1, log_6_1 = await self.specific_format(
        subtask_id="stage_6.subtask_1",
        formatter_desc=formatter_desc_6_1
    )
    logs.append(log_6_1)
    results["stage_6.subtask_1"] = results_6_1

    final_answer = await self.make_final_answer(results_6_1["thinking"], results_6_1["answer"])
    return final_answer, logs
