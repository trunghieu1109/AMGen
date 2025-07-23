async def forward_153(self, taskInfo):
    logs = []

    cot_instruction_stage0_subtask1 = (
        "Sub-task 1: Extract and summarize key spectral features from mass, IR, and 1H NMR data provided in the query. "
        "Input: taskInfo containing question and spectral data."
    )
    cot_agent_desc_stage0_subtask1 = {
        "instruction": cot_instruction_stage0_subtask1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_stage0_subtask1, log_stage0_subtask1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_subtask1
    )
    logs.append(log_stage0_subtask1)

    cot_instruction_stage1_subtask1 = (
        "Sub-task 1: Analyze spectral features to classify and assign possible structural categories to candidates. "
        "Input: results (thinking and answer) from stage_0.subtask_1 and taskInfo."
    )
    cot_agent_desc_stage1_subtask1 = {
        "instruction": cot_instruction_stage1_subtask1,
        "input": [taskInfo, results_stage0_subtask1['thinking'], results_stage0_subtask1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_subtask1, log_stage1_subtask1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_stage1_subtask1
    )
    logs.append(log_stage1_subtask1)

    loop_results = {
        "stage_2.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_stage2_subtask1 = (
            f"Iteration {iteration+1} - Sub-task 1: Generate initial reasoning steps linking spectral data to candidate structures. "
            "Input: results (thinking and answer) from stage_0.subtask_1 and taskInfo."
        )
        cot_agent_desc_stage2_subtask1 = {
            "instruction": cot_instruction_stage2_subtask1,
            "input": [taskInfo, results_stage0_subtask1['thinking'], results_stage0_subtask1['answer']],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_stage2_subtask1, log_stage2_subtask1 = await self.cot(
            subtask_id=f"stage_2.subtask_1.iter{iteration+1}",
            cot_agent_desc=cot_agent_desc_stage2_subtask1
        )
        logs.append(log_stage2_subtask1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_stage2_subtask1['thinking'])
        loop_results["stage_2.subtask_1"]["answer"].append(results_stage2_subtask1['answer'])

        cot_instruction_stage2_subtask2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine and consolidate reasoning to improve structural suggestion accuracy. "
            "Input: all previous results (thinking and answer) from stage_2.subtask_1 iterations and results from stage_1.subtask_1, plus taskInfo."
        )
        cot_agent_desc_stage2_subtask2 = {
            "instruction": cot_instruction_stage2_subtask2,
            "input": [taskInfo] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"] + [results_stage1_subtask1['thinking'], results_stage1_subtask1['answer']],
            "temperature": 0.0,
            "context_desc": [
                "user query",
                "all thinking of stage_2.subtask_1 iterations",
                "all answers of stage_2.subtask_1 iterations",
                "thinking of stage_1.subtask_1",
                "answer of stage_1.subtask_1"
            ]
        }
        results_stage2_subtask2, log_stage2_subtask2 = await self.reflexion(
            subtask_id=f"stage_2.subtask_2.iter{iteration+1}",
            reflect_desc=cot_agent_desc_stage2_subtask2,
            n_repeat=1
        )
        logs.append(log_stage2_subtask2)
        loop_results["stage_2.subtask_2"]["thinking"].append(results_stage2_subtask2['thinking'])
        loop_results["stage_2.subtask_2"]["answer"].append(results_stage2_subtask2['answer'])

    cot_instruction_stage3_subtask1 = (
        "Sub-task 1: Evaluate refined reasoning and select the most plausible structural candidate from given choices. "
        "Input: results (thinking and answer) from stage_1.subtask_1 and all iterations of stage_2.subtask_2, plus taskInfo."
    )
    cot_agent_desc_stage3_subtask1 = {
        "instruction": cot_instruction_stage3_subtask1,
        "input": [taskInfo, results_stage1_subtask1['thinking'], results_stage1_subtask1['answer']] + loop_results["stage_2.subtask_2"]["thinking"] + loop_results["stage_2.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "all thinking of stage_2.subtask_2 iterations",
            "all answers of stage_2.subtask_2 iterations"
        ]
    }
    results_stage3_subtask1, log_stage3_subtask1 = await self.answer_generate(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_stage3_subtask1
    )
    logs.append(log_stage3_subtask1)

    final_answer = await self.make_final_answer(results_stage3_subtask1['thinking'], results_stage3_subtask1['answer'])
    return final_answer, logs
