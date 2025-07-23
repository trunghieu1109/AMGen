async def forward_165(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the given Lagrangian and particle content to identify relevant mass terms "
        "and their contributions to the pseudo-Goldstone boson mass formula. Input: taskInfo containing the query with Lagrangian, particle content, and VEVs."
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

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Identify and list all particle mass contributions and coefficients "
            "relevant for the radiative correction formula. Input: taskInfo, results from stage_0.subtask_1, and all previous results from stage_2.subtask_1 iterations."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_1['thinking']] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1"] + ["thinking of previous stage_2.subtask_1 iterations"]*len(loop_results["stage_2.subtask_1"]["thinking"])
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_2_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Compute the approximate mass squared of the pseudo-Goldstone boson "
            "using the constructed intermediate terms and evaluate candidate formulas. Input: taskInfo and all results from stage_1.subtask_1 iterations."
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query"] + ["thinking of stage_1.subtask_1 iterations"]*len(loop_results["stage_1.subtask_1"]["thinking"])
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

    debate_instruction_3_1 = (
        "Sub-task 1: Compare candidate mass formulas against theoretical expectations and select the best approximation "
        "for the pseudo-Goldstone boson mass. Input: taskInfo, all results from stage_1.subtask_1 and stage_2.subtask_1 iterations."
    )
    debate_final_decision_3_1 = (
        "Sub-task 1: Select the best approximation formula for the pseudo-Goldstone boson mass based on the comparison."
    )
    debate_desc_3_1 = {
        "instruction": debate_instruction_3_1,
        "final_decision_instruction": debate_final_decision_3_1,
        "input": [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
        "context_desc": ["user query"] + ["thinking of stage_1.subtask_1 iterations"]*len(loop_results["stage_1.subtask_1"]["thinking"]) + ["answer of stage_1.subtask_1 iterations"]*len(loop_results["stage_1.subtask_1"]["answer"]) + ["thinking of stage_2.subtask_1 iterations"]*len(loop_results["stage_2.subtask_1"]["thinking"]) + ["answer of stage_2.subtask_1 iterations"]*len(loop_results["stage_2.subtask_1"]["answer"]),
        "temperature": 0.5
    }
    results_3_1, log_3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc_3_1,
        n_repeat=1
    )
    logs.append(log_3_1)

    reflexion_instruction_4_1 = (
        "Sub-task 1: Apply necessary transformations to the selected formula and validate its consistency "
        "with model constraints and physical principles. Input: taskInfo and results from stage_3.subtask_1."
    )
    critic_instruction_4_1 = (
        "Please review and provide the limitations of the selected formula and its transformations."
    )
    reflexion_desc_4_1 = {
        "instruction": reflexion_instruction_4_1,
        "critic_instruction": critic_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_4_1, log_4_1 = await self.reflexion(
        subtask_id="stage_4.subtask_1",
        reflect_desc=reflexion_desc_4_1,
        n_repeat=2
    )
    logs.append(log_4_1)

    review_instruction_5_1 = (
        "Sub-task 1: Evaluate the final mass formula for correctness, consistency, and compliance with the problem statement. "
        "Input: taskInfo, results from stage_3.subtask_1 and stage_4.subtask_1."
    )
    review_desc_5_1 = {
        "instruction": review_instruction_5_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"], results_4_1["thinking"], results_4_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"]
    }
    results_5_1, log_5_1 = await self.review(
        subtask_id="stage_5.subtask_1",
        review_desc=review_desc_5_1
    )
    logs.append(log_5_1)

    final_answer = await self.make_final_answer(results_5_1["thinking"], results_5_1["answer"])
    return final_answer, logs
