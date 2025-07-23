async def forward_197(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and summarize all given information from the query, including concentrations and stability constants."
    )
    cot_agent_desc_0_0 = {
        "instruction": cot_instruction_0_0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_0, log_0_0 = await self.cot(
        subtask_id="stage_0.subtask_0",
        cot_agent_desc=cot_agent_desc_0_0
    )
    logs.append(log_0_0)

    cot_instruction_0_1 = (
        "Sub-task 1: Formulate the equilibrium expressions for all cobalt(II) thiocyanato complexes using the cumulative stability constants and free SCN- concentration, based on the extracted information."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo, results_0_0["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_0"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Set up the mass balance equation for total cobalt concentration distributed among free Co(II) and all complex species, using the equilibrium expressions."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Solve the mass balance equation to find the free Co(II) concentration and concentrations of each complex species."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Calculate the concentration of the blue dithiocyanato complex Co(SCN)2 from the solved species concentrations."
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_3["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    aggregate_instruction_1_0 = (
        "Sub-task 0: Combine the calculated concentrations of all cobalt species to verify total cobalt balance."
    )
    aggregate_desc_1_0 = {
        "instruction": aggregate_instruction_1_0,
        "input": [taskInfo, results_0_4["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_4"]
    }
    results_1_0, log_1_0 = await self.aggregate(
        subtask_id="stage_1.subtask_0",
        aggregate_desc=aggregate_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Compute the percentage of the blue dithiocyanato complex relative to total cobalt concentration."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_agent_instruction_1_1,
        "input": [taskInfo, results_1_0["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_2_0 = (
        "Sub-task 0: Validate the calculated percentage against the given multiple-choice options for plausibility and correctness."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_1["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2.subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    aggregate_instruction_2_1 = (
        "Sub-task 1: Select the closest matching answer choice based on the calculated percentage."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo, results_2_0["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0"]
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id="stage_2.subtask_1",
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the selected answer considering assumptions and possible errors in the calculation."
    )
    final_decision_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the selected answer considering assumptions and possible errors in the calculation."
    )
    debate_desc_2_2 = {
        "instruction": debate_instruction_2_2,
        "final_decision_instruction": final_decision_instruction_2_2,
        "input": [taskInfo, results_2_1["thinking"]],
        "context_desc": ["user query", "thinking of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    formatter_instruction_3_0 = (
        "Sub-task 0: Format the final answer and explanation into a clear, concise summary suitable for presentation."
    )
    formatter_desc_3_0 = {
        "instruction": formatter_instruction_3_0,
        "input": [taskInfo, results_2_2["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2"],
        "format": "short and concise, without explaination"
    }
    results_3_0, log_3_0 = await self.specific_format(
        subtask_id="stage_3.subtask_0",
        formatter_desc=formatter_desc_3_0
    )
    logs.append(log_3_0)

    review_instruction_3_1 = (
        "Sub-task 1: Review the formatted output for clarity, correctness, and completeness."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_3_0["thinking"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_0"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])

    return final_answer, logs
