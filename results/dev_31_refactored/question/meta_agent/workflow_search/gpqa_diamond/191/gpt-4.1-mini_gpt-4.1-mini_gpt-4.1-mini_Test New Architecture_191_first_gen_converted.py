async def forward_191(self, taskInfo):
    logs = []
    results = {}

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize all given physical parameters and geometric relationships from the problem statement. "
        "Input content: taskInfo"
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

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the electrostatic interactions, including induced charges and vector relationships between cavity, conductor, and point P. "
        "Input content: results from stage_0.subtask_1 (thinking and answer) and taskInfo"
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
    results["stage_0.subtask_2"] = results_0_2

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Formulate the expression for the electric field at point P considering the charge inside the cavity and conductor shielding effects. "
            f"Input content: results from stage_0.subtask_2 and all previous iterations of stage_1.subtask_2 (thinking and answer), plus taskInfo"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + [f"thinking of stage_1.subtask_2 iteration {i+1}" for i in range(len(loop_results["stage_1.subtask_2"]["thinking"]))] + [f"answer of stage_1.subtask_2 iteration {i+1}" for i in range(len(loop_results["stage_1.subtask_2"]["answer"]))]
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id=f"stage_1.subtask_1.iter{iteration+1}",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_agent_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine the intermediate expression by incorporating vector geometry and angle theta dependencies. "
            f"Input content: results from stage_1.subtask_1 (thinking and answer)"
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_agent_instruction_1_2,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", f"thinking of stage_1.subtask_1 iteration {iteration+1}", f"answer of stage_1.subtask_1 iteration {iteration+1}"]
        }
        results_1_2, log_1_2 = await self.answer_generate(
            subtask_id=f"stage_1.subtask_2.iter{iteration+1}",
            cot_agent_desc=cot_agent_desc_1_2
        )
        logs.append(log_1_2)
        loop_results["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    cot_instruction_2_1 = (
        "Sub-task 1: Evaluate the intermediate expressions and select the most physically consistent formula for the electric field magnitude at point P. "
        "Input content: results from stage_1.subtask_1 and stage_1.subtask_2 (thinking and answer) from all iterations, plus taskInfo"
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query"] + [f"thinking of stage_1.subtask_1 iteration {i+1}" for i in range(2)] + [f"answer of stage_1.subtask_1 iteration {i+1}" for i in range(2)] + [f"thinking of stage_1.subtask_2 iteration {i+1}" for i in range(2)] + [f"answer of stage_1.subtask_2 iteration {i+1}" for i in range(2)]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)
    results["stage_2.subtask_1"] = results_2_1

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Check the selected expression against boundary conditions and physical constraints to confirm validity. "
        "Input content: results from stage_2.subtask_1 (thinking and answer)"
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Synthesize and choose the most consistent and valid expression for the electric field magnitude at point P."
    )
    cot_sc_desc_3_1 = {
        "instruction": cot_sc_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc_3_1,
        n_repeat=self.max_sc if hasattr(self, 'max_sc') else 3
    )
    logs.append(log_3_1)
    results["stage_3.subtask_1"] = results_3_1

    formatter_instruction_4_1 = (
        "Sub-task 1: Present the final electric field magnitude expression in a clear, concise format matching the problem's notation. "
        "Input content: results from stage_3.subtask_1 (thinking and answer)"
    )
    formatter_desc_4_1 = {
        "instruction": formatter_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "format": "short and concise, without explanation"
    }
    results_4_1, log_4_1 = await self.specific_format(
        subtask_id="stage_4.subtask_1",
        formatter_desc=formatter_desc_4_1
    )
    logs.append(log_4_1)
    results["stage_4.subtask_1"] = results_4_1

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
