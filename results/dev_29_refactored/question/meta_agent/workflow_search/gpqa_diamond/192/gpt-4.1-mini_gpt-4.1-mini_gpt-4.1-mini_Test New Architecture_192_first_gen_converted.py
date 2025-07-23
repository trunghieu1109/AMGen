async def forward_192(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": []}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize the given information from the query, including the relationship between number of stars and parallax, "
            "and the inverse relation between parallax and distance."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id=f"stage_0.subtask_0.iter{iteration}",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        stage0_results["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Express the number of stars as a function of distance by substituting parallax with its inverse proportionality to distance."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0["thinking"] if "thinking" in results_0_0 else results_0_0],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id=f"stage_0.subtask_1.iter{iteration}",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        stage0_results["subtask_1"].append(results_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Determine the differential relationship to find the number of stars per unit distance range, "
            "considering the change of variables from parallax to distance."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_1["thinking"] if "thinking" in results_0_1 else results_0_1],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id=f"stage_0.subtask_2.iter{iteration}",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        stage0_results["subtask_2"].append(results_0_2)

        formatter_instruction_0_3 = (
            "Sub-task 3: Document the intermediate mathematical expressions and reasoning steps clearly to support further refinement."
        )
        formatter_desc_0_3 = {
            "instruction": formatter_instruction_0_3,
            "input": [taskInfo, results_0_2["thinking"] if "thinking" in results_0_2 else results_0_2],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2"],
            "format": "short and concise, without explaination"
        }
        results_0_3, log_0_3 = await self.specific_format(
            subtask_id=f"stage_0.subtask_3.iter{iteration}",
            formatter_desc=formatter_desc_0_3
        )
        logs.append(log_0_3)
        stage0_results["subtask_3"].append(results_0_3)

    cot_instruction_1_0 = (
        "Sub-task 0: Simplify and consolidate the intermediate expressions to a clear power-law form relating number of stars per unit distance to distance r."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_instruction_1_0,
        "input": [taskInfo] + stage0_results["subtask_3"],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from stage_0.subtask_3"]
    }
    results_1_0, log_1_0 = await self.cot(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Evaluate the candidate power-law exponents against the provided multiple-choice options to select the best matching answer."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_agent_instruction_1_1,
        "input": [taskInfo, results_1_0["thinking"] if "thinking" in results_1_0 else results_1_0],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_2_0 = (
        "Sub-task 0: Apply the final transformation and substitution to confirm the selected power-law relationship and prepare the final answer format."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_1["thinking"] if "thinking" in results_1_1 else results_1_1],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2.subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    review_instruction_3_0 = (
        "Sub-task 0: Validate the correctness and consistency of the derived relationship and the selected answer choice with astrophysical principles and problem constraints."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results_2_0["thinking"] if "thinking" in results_2_0 else results_2_0],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0"]
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id="stage_3.subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    final_answer = await self.make_final_answer(
        results_3_0.get("thinking", ""),
        results_1_1.get("answer", "")
    )

    return final_answer, logs
