async def forward_172(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract given values (electron speed v, position uncertainty Δx) and convert units as needed. "
        "Input: taskInfo containing question and choices."
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

    cot_instruction_0_2 = (
        "Sub-task 2: Calculate the minimum uncertainty in momentum Δp using the Heisenberg uncertainty principle Δx·Δp ≥ ħ/2. "
        "Input: results (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Relate uncertainty in momentum Δp to uncertainty in kinetic energy ΔE using E = p²/(2m) and propagate uncertainty. "
        "Input: results (thinking and answer) from stage_0.subtask_2."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for uncertainty in energy ΔE estimation."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    loop_results = {
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_2_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Apply systematic procedure to refine ΔE estimate based on previous analysis and assumptions. "
            f"Input: results (thinking and answer) from stage_1.subtask_1 and all previous iterations of this subtask."
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"] + [f"thinking iteration {i+1}" for i in range(iteration)] + [f"answer iteration {i+1}" for i in range(iteration)]
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id=f"stage_2.subtask_1_iter_{iteration+1}",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

    cot_instruction_3_1 = (
        "Sub-task 1: Compare refined ΔE estimate with given choices and select the closest matching option. "
        "Input: results (thinking and answer) from all iterations of stage_2.subtask_1."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query"] + [f"thinking iteration {i+1}" for i in range(2)] + [f"answer iteration {i+1}" for i in range(2)]
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
