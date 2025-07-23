async def forward_186(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and summarize all given information about the stars, instrument, and detectability criteria from the query."
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
        "Sub-task 1: Calculate the apparent V magnitude for each star using the absolute magnitude and distance (distance modulus formula)."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Estimate the expected signal-to-noise ratio (S/N) for each star based on its apparent magnitude, the ESPRESSO spectrograph sensitivity, telescope aperture, and 1-hour exposure time."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Document assumptions and reasoning used in the calculations, including any approximations about instrument sensitivity and star spectral types."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    aggregate_instruction_1_0 = (
        "Sub-task 0: Combine the calculated S/N values and star data into a consolidated dataset for evaluation."
    )
    aggregate_desc_1_0 = {
        "instruction": aggregate_instruction_1_0,
        "input": [taskInfo, results_0_3["thinking"], results_0_3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from stage_0.subtask_3"]
    }
    results_1_0, log_1_0 = await self.aggregate(
        subtask_id="stage_1.subtask_0",
        aggregate_desc=aggregate_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Apply the detectability criterion (S/N ≥ 10) to each star to classify it as detectable or not."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_agent_instruction_1_1,
        "input": [taskInfo, results_1_0["thinking"], results_1_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    review_instruction_2_0 = (
        "Sub-task 0: Validate the detectability results by cross-checking with known brightness of Canopus and Polaris to ensure consistency."
    )
    review_desc_2_0 = {
        "instruction": review_instruction_2_0,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.review(
        subtask_id="stage_2.subtask_0",
        review_desc=review_desc_2_0
    )
    logs.append(log_2_0)

    aggregate_instruction_2_1 = (
        "Sub-task 1: Select the stars that satisfy the detectability condition and count how many meet the criterion."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo, results_2_0["thinking"], results_2_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"]
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id="stage_2.subtask_1",
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the final count considering assumptions and possible uncertainties."
    )
    final_decision_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the final count considering assumptions and possible uncertainties."
    )
    debate_desc_2_2 = {
        "instruction": debate_instruction_2_2,
        "final_decision_instruction": final_decision_instruction_2_2,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    formatter_instruction_3_0 = (
        "Sub-task 0: Format the final answer by summarizing the number of detectable stars and providing a brief explanation."
    )
    formatter_desc_3_0 = {
        "instruction": formatter_instruction_3_0,
        "input": [taskInfo, results_2_2["thinking"], results_2_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        "format": "short and concise, without explaination"
    }
    results_3_0, log_3_0 = await self.specific_format(
        subtask_id="stage_3.subtask_0",
        formatter_desc=formatter_desc_3_0
    )
    logs.append(log_3_0)

    cot_agent_instruction_3_1 = (
        "Sub-task 1: Map the final count to the provided multiple-choice options and select the best matching choice."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo, results_3_0["thinking"], results_3_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_0", "answer of stage_3.subtask_0"]
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])

    return final_answer, logs
