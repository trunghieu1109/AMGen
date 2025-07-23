async def forward_185(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Stage 0 - Subtask 1: Extract and summarize the starting compound, reaction type, stereochemistry, and possible product options from the query. "
        "Input: taskInfo containing question and choices."
    )
    cot_agent_desc_stage0 = {
        "instruction": cot_instruction_stage0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_stage0_subtask1, log_stage0_subtask1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0
    )
    logs.append(log_stage0_subtask1)

    loop_results_stage1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []},
        "stage_1.subtask_3": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Stage 1 - Iteration {iteration+1} - Subtask 1: Construct intermediate mechanistic steps and possible rearrangement pathways for the Cope rearrangement of the given bicyclic compound. "
            "Input: taskInfo and outputs (thinking and answer) from stage_0.subtask_1."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_stage0_subtask1["thinking"], results_stage0_subtask1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results_stage1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_1_2 = (
            f"Stage 1 - Iteration {iteration+1} - Subtask 2: Analyze stereochemical and regiochemical relationships influencing product formation and classify possible product isomers. "
            "Input: taskInfo, outputs from stage_0.subtask_1, and all previous outputs from stage_1.subtask_1 iterations."
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_instruction_1_2,
            "input": [taskInfo, results_stage0_subtask1["thinking"], results_stage0_subtask1["answer"]] + loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "all thinkings of stage_1.subtask_1", "all answers of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_agent_desc_1_2
        )
        logs.append(log_1_2)
        loop_results_stage1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

        cot_instruction_1_3 = (
            f"Stage 1 - Iteration {iteration+1} - Subtask 3: Derive and refine the most plausible product structure(s) consistent with the Cope rearrangement and stereochemical constraints. "
            "Input: taskInfo and all outputs from stage_1.subtask_1 and stage_1.subtask_2 iterations."
        )
        cot_agent_desc_1_3 = {
            "instruction": cot_instruction_1_3,
            "input": [taskInfo] + loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "all thinkings of stage_1.subtask_1", "all answers of stage_1.subtask_1", "all thinkings of stage_1.subtask_2", "all answers of stage_1.subtask_2"]
        }
        results_1_3, log_1_3 = await self.cot(
            subtask_id="stage_1.subtask_3",
            cot_agent_desc=cot_agent_desc_1_3
        )
        logs.append(log_1_3)
        loop_results_stage1["stage_1.subtask_3"]["thinking"].append(results_1_3["thinking"])
        loop_results_stage1["stage_1.subtask_3"]["answer"].append(results_1_3["answer"])

    cot_instruction_stage2 = (
        "Stage 2 - Subtask 1: Evaluate the derived product candidates against the query's product options and select the best matching product. "
        "Input: taskInfo and all outputs from stage_1.subtask_3 iterations."
    )
    cot_agent_desc_stage2 = {
        "instruction": cot_instruction_stage2,
        "input": [taskInfo] + loop_results_stage1["stage_1.subtask_3"]["thinking"] + loop_results_stage1["stage_1.subtask_3"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "all thinkings of stage_1.subtask_3", "all answers of stage_1.subtask_3"]
    }
    results_stage2_subtask1, log_stage2_subtask1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_stage2
    )
    logs.append(log_stage2_subtask1)

    final_answer = await self.make_final_answer(results_stage2_subtask1["thinking"], results_stage2_subtask1["answer"])
    return final_answer, logs
