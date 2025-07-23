async def forward_195(self, taskInfo):
    logs = []
    stage0_results = {}
    stage1_2_loop_results = {"stage_1.subtask_1": {"thinking": [], "answer": []}, "stage_2.subtask_1": {"thinking": [], "answer": []}}

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize given physical information and parameters from the query. "
        "Input: the query containing the question and candidate formulas."
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
    stage0_results["stage_0.subtask_1"] = results_0_1

    cot_instruction_0_2 = (
        "Sub-task 2: Parse and represent the candidate formulas for maximum speed in a structured form. "
        "Input: the query and the summarized physical information from Sub-task 1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)
    stage0_results["stage_0.subtask_2"] = results_0_2

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Analyze the physical consistency and relativistic validity of each candidate formula. "
            "Input: the query, parsed candidate formulas from stage_0.subtask_2, and all previous iteration results of stage_2.subtask_1."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, stage0_results["stage_0.subtask_2"]["thinking"], stage0_results["stage_0.subtask_2"]["answer"]] + stage1_2_loop_results["stage_2.subtask_1"]["thinking"] + stage1_2_loop_results["stage_2.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["thinking of previous stage_2.subtask_1 iterations"]*len(stage1_2_loop_results["stage_2.subtask_1"]["thinking"])
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        stage1_2_loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        stage1_2_loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        aggregate_instruction_2_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Validate candidate formulas against relativistic constraints and select the physically plausible maximum speed expression. "
            "Input: the query and all analysis results from stage_1.subtask_1 iterations so far."
        )
        aggregate_desc_2_1 = {
            "instruction": aggregate_instruction_2_1,
            "input": [taskInfo] + stage1_2_loop_results["stage_1.subtask_1"]["answer"] + stage1_2_loop_results["stage_1.subtask_1"]["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query", "answers and thinkings from stage_1.subtask_1 iterations"]
        }
        results_2_1, log_2_1 = await self.aggregate(
            subtask_id="stage_2.subtask_1",
            aggregate_desc=aggregate_desc_2_1
        )
        logs.append(log_2_1)
        stage1_2_loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        stage1_2_loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

    cot_instruction_3_1 = (
        "Sub-task 1: Consolidate evaluation results and format the final selected formula for maximum speed. "
        "Input: the query and all validation results from stage_2.subtask_1 iterations."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo] + stage1_2_loop_results["stage_2.subtask_1"]["thinking"] + stage1_2_loop_results["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinkings of stage_2.subtask_1 iterations", "answers of stage_2.subtask_1 iterations"]
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
