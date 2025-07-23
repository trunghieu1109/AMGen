async def forward_168(self, taskInfo):
    logs = []
    
    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the original decay process 2A -> 2B + 2E + 2V and characterize the energy spectrum of emitted E particles, "
        "including the role of emitted V particles and the endpoint Q. Input: [taskInfo]"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the variant decay process replacing 2V with a single massless M particle (2A -> 2B + 2E + M) and predict qualitative effects on the energy spectrum continuity and endpoint. "
        "Input: [taskInfo, thinking and answer from stage_0.subtask_1]"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Generate initial intermediate reasoning on how the replacement of 2V by M affects phase space and energy distribution of E particles. "
            f"Input: [taskInfo, all previous thinking and answers from stage_0.subtask_2 and all previous iterations of stage_1.subtask_1 and stage_1.subtask_2]"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + loop_results_stage_1["stage_1.subtask_2"]["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["answer of previous iterations of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["answer"]) + ["thinking of previous iterations of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]) + ["answer of previous iterations of stage_1.subtask_2"]*len(loop_results_stage_1["stage_1.subtask_2"]["answer"]) + ["thinking of previous iterations of stage_1.subtask_2"]*len(loop_results_stage_1["stage_1.subtask_2"]["thinking"])
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine intermediate reasoning by considering conservation laws and particle mass effects on spectrum shape and endpoint, incorporating feedback from previous iteration. "
            f"Input: [taskInfo, thinking and answer from stage_1.subtask_1 (all iterations)]"
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_instruction_1_2,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query"] + ["thinking of all iterations of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]) + ["answer of all iterations of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["answer"])
        }
        results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    debate_instruction_2_1 = (
        "Sub-task 1: Evaluate candidate answers against derived reasoning and select the best matching explanation for the spectrum continuity and endpoint shift. "
        "Input: [taskInfo, thinking and answer from stage_0.subtask_2, and thinking and answer from all iterations of stage_1.subtask_2]"
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": "Sub-task 1: Select the best explanation for the spectrum continuity and endpoint shift based on all prior reasoning.",
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["thinking of all iterations of stage_1.subtask_2"]*len(loop_results_stage_1["stage_1.subtask_2"]["thinking"]) + ["answer of all iterations of stage_1.subtask_2"]*len(loop_results_stage_1["stage_1.subtask_2"]["answer"]),
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(subtask_id="stage_2.subtask_1", debate_desc=debate_desc_2_1, n_repeat=self.max_round)
    logs.append(log_2_1)

    aggregate_instruction_3_1 = (
        "Sub-task 1: Critically assess the selected candidate for physical correctness and consistency, refining the explanation as needed. "
        "Input: [taskInfo, thinking and answer from stage_2.subtask_1]"
    )
    aggregate_desc_3_1 = {
        "instruction": aggregate_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.aggregate(subtask_id="stage_3.subtask_1", aggregate_desc=aggregate_desc_3_1)
    logs.append(log_3_1)

    formatter_instruction_4_1 = (
        "Sub-task 1: Consolidate the refined explanation into a clear, concise final answer matching the query format. "
        "Input: [taskInfo, thinking and answer from stage_3.subtask_1]"
    )
    formatter_desc_4_1 = {
        "instruction": formatter_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "format": "short and concise, without explanation"
    }
    results_4_1, log_4_1 = await self.specific_format(subtask_id="stage_4.subtask_1", formatter_desc=formatter_desc_4_1)
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
