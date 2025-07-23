async def forward_188(self, taskInfo):
    logs = []
    results = {}

    cot_instruction_0_1 = (
        "Subtask stage_0.subtask_1: Identify and extract the list of effective particles and the key condition about spontaneous symmetry breaking from the query. "
        "Input: taskInfo containing the question and choices."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)
    results["stage_0.subtask_1"] = results_0_1

    cot_instruction_1_1 = (
        "Subtask stage_1.subtask_1: Analyze each particle's physical origin and classify whether it is associated with spontaneously-broken symmetry or not. "
        "Input: results (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)
    results["stage_1.subtask_1"] = results_1_1

    loop_results_stage_2 = {"thinking": [], "answer": []}
    for iteration in range(2):
        cot_instruction_2_1 = (
            f"Iteration {iteration+1} of subtask stage_2.subtask_1: For each particle, iteratively refine reasoning about its connection to spontaneous symmetry breaking, "
            "using prior iteration insights. Input: results (thinking and answer) from stage_1.subtask_1 and all previous iterations of stage_2.subtask_1."
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]] + loop_results_stage_2["answer"] + loop_results_stage_2["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"] + [f"answer iteration {i+1} of stage_2.subtask_1" for i in range(len(loop_results_stage_2["answer"]))] + [f"thinking iteration {i+1} of stage_2.subtask_1" for i in range(len(loop_results_stage_2["thinking"]))]
        }
        results_2_1, log_2_1 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2_1)
        logs.append(log_2_1)
        loop_results_stage_2["thinking"].append(results_2_1["thinking"])
        loop_results_stage_2["answer"].append(results_2_1["answer"])
    results["stage_2.subtask_1"] = {
        "thinking": loop_results_stage_2["thinking"],
        "answer": loop_results_stage_2["answer"]
    }

    cot_instruction_3_1 = (
        "Subtask stage_3.subtask_1: Combine the iterative reasoning outcomes to form a consolidated list indicating which particles are associated or not with spontaneously-broken symmetry. "
        "Input: results (thinking and answer) from all iterations of stage_2.subtask_1."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo] + loop_results_stage_2["answer"] + loop_results_stage_2["thinking"],
        "temperature": 0.0,
        "context_desc": ["user query"] + [f"answer iteration {i+1} of stage_2.subtask_1" for i in range(len(loop_results_stage_2["answer"]))] + [f"thinking iteration {i+1} of stage_2.subtask_1" for i in range(len(loop_results_stage_2["thinking"]))]
    }
    results_3_1, log_3_1 = await self.cot(subtask_id="stage_3.subtask_1", cot_agent_desc=cot_agent_desc_3_1)
    logs.append(log_3_1)
    results["stage_3.subtask_1"] = results_3_1

    cot_instruction_4_1 = (
        "Subtask stage_4.subtask_1: Evaluate the consolidated list to select the particle not associated with spontaneously-broken symmetry. "
        "Input: results (thinking and answer) from stage_3.subtask_1."
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_4_1, log_4_1 = await self.cot(subtask_id="stage_4.subtask_1", cot_agent_desc=cot_agent_desc_4_1)
    logs.append(log_4_1)
    results["stage_4.subtask_1"] = results_4_1

    cot_agent_instruction_5_1 = (
        "Subtask stage_5.subtask_1: Format the selected answer into a clear, concise final output. "
        "Input: results (thinking and answer) from stage_4.subtask_1."
    )
    cot_agent_desc_5_1 = {
        "instruction": cot_agent_instruction_5_1,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"]
    }
    results_5_1, log_5_1 = await self.answer_generate(subtask_id="stage_5.subtask_1", cot_agent_desc=cot_agent_desc_5_1)
    logs.append(log_5_1)
    results["stage_5.subtask_1"] = results_5_1

    final_answer = await self.make_final_answer(results_5_1["thinking"], results_5_1["answer"])
    return final_answer, logs
