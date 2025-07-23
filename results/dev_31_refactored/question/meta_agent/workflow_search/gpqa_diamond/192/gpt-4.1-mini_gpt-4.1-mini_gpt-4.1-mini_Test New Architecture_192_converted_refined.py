async def forward_192(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}, "stage_0.subtask_3": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Identify and extract key elements from the query: star count dependence on parallax, "
            "parallax-distance relation, and problem goal, explicitly clarifying whether the given relation 'number of stars ∝ 1/plx^5' "
            "refers to a differential density (dN/dplx) or a cumulative count N(<plx). This is critical to avoid the previous error of misinterpreting differential as cumulative, "
            "which led to incorrect extra differentiation. Input content: user query."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])
        logs.append(log_0_1)

        cot_sc_instruction_0_2 = (
            "Sub-task 2: Validate and self-critique the interpretation from subtask_1 by checking physical and mathematical consistency of differential vs cumulative interpretations, "
            "ensuring the chosen interpretation aligns with astrophysical reasoning and prevents the previous failure of incorrect scaling. "
            "Input content: results (thinking and answer) from stage_0.subtask_1."
        )
        final_decision_instruction_0_2 = (
            "Sub-task 2: Synthesize and choose the most consistent interpretation for the star count relation with parallax, "
            "based on physical and mathematical reasoning."
        )
        cot_sc_desc_0_2 = {
            "instruction": cot_sc_instruction_0_2,
            "final_decision_instruction": final_decision_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.sc_cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_sc_desc_0_2, n_repeat=self.max_sc)
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Summarize and clearly state the mathematical relationships and constraints, including the parallax-distance inverse proportionality and the confirmed interpretation of star count variation, "
            "preparing for substitution and transformation in the next stage. Input content: results (thinking and answer) from stage_0.subtask_2."
        )
        cot_agent_desc_0_3 = {
            "instruction": cot_instruction_0_3,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_0_3, log_0_3 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_0_3)
        loop_results["stage_0.subtask_3"]["thinking"].append(results_0_3["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results_0_3["answer"])
        logs.append(log_0_3)

    cot_instruction_1_1 = (
        "Sub-task 1: Substitute the parallax-distance relation (plx ∝ 1/r) into the star count formula (dN/dplx ∝ 1/plx^5) using the confirmed differential interpretation, "
        "and derive the expression for star count variation as a function of distance r. Input content: results (thinking and answer) from stage_0.subtask_3."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Interpret the resulting mathematical expression to determine how the number of stars per unit distance interval varies with distance r, "
        "explicitly incorporating the derivative |d(plx)/dr| and avoiding the previous mistake of unnecessary extra differentiation. "
        "Input content: results (thinking and answer) from stage_1.subtask_1."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Match the derived star count variation with distance r to the given answer choices, "
        "providing a clear justification for the selection based on the previous mathematical reasoning and interpretations. "
        "Input content: results (thinking and answer) from stage_1.subtask_2."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1["thinking"], results_2_1["answer"])
    return final_answer, logs
