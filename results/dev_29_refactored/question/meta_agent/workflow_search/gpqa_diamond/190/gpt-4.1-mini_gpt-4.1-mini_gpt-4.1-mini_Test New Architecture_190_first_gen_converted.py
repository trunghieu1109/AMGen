async def forward_190(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and summarize the given chemical information and reaction steps from the query to form a clear understanding of the starting material and each transformation."
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
        "Sub-task 1: Analyze the chemical transformations stepwise, predicting the structure of each intermediate product (products 1 to 4) based on known reaction mechanisms, using the summary from Sub-task 0."
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

    review_instruction_0_2 = (
        "Sub-task 2: Document the reasoning and structural changes at each step, including the fate of functional groups and substituents, based on the analysis from Sub-task 1."
    )
    review_desc_0_2 = {
        "instruction": review_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.review(
        subtask_id="stage_0.subtask_2",
        review_desc=review_desc_0_2
    )
    logs.append(log_0_2)

    aggregate_instruction_1_0 = (
        "Sub-task 0: Combine the intermediate product structures and mechanistic insights into a consolidated reaction pathway summary, using outputs from stage_0.subtask_2."
    )
    aggregate_desc_1_0 = {
        "instruction": aggregate_instruction_1_0,
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from stage_0.subtask_2"]
    }
    results_1_0, log_1_0 = await self.aggregate(
        subtask_id="stage_1.subtask_0",
        aggregate_desc=aggregate_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Evaluate the consolidated pathway against the multiple-choice options to identify which product structure matches the final product 4, using the summary from stage_1.subtask_0."
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

    cot_instruction_2_0 = (
        "Sub-task 0: Validate the chemical plausibility of the selected product by cross-checking reaction conditions and typical outcomes of each step, based on the answer from stage_1.subtask_1."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2.subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    debate_instruction_2_1 = (
        "Sub-task 1: Select the most chemically valid product option from the multiple choices based on the validation from stage_2.subtask_0."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Select the most chemically valid product option from the multiple choices based on the validation."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_2_0["thinking"], results_2_0["answer"]],
        "context": ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    review_instruction_2_2 = (
        "Sub-task 2: Assess the final selection for consistency with the entire reaction sequence and mechanistic rationale, based on the debate results from stage_2.subtask_1."
    )
    review_desc_2_2 = {
        "instruction": review_instruction_2_2,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.review(
        subtask_id="stage_2.subtask_2",
        review_desc=review_desc_2_2
    )
    logs.append(log_2_2)

    formatter_instruction_3_0 = (
        "Sub-task 0: Consolidate the validated product structure and reasoning into a clear, concise final answer format, based on stage_2.subtask_2."
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

    review_instruction_3_1 = (
        "Sub-task 1: Summarize the entire solution process, highlighting key mechanistic insights and final product identification, based on the formatted final answer from stage_3.subtask_0."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_3_0["thinking"], results_3_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_0", "answer of stage_3.subtask_0"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
