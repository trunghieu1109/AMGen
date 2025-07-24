async def forward_28(self, taskInfo):
    logs = []

    # Stage 0, Sub-task 1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Determine the parametric and geometric formulas for the torus and sphere, including the radii and center positions relevant to tangency. "
        "Input content are results (both thinking and answer) from: none."
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

    # Stage 0, Sub-task 2
    cot_instruction_0_2 = (
        "Stage 0, Sub-task 2: Formulate expressions for the radii of the tangent circles (r_i and r_o) when the torus rests externally tangent to the sphere in two configurations. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
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

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for i in range(3):
        cot_sc_instruction_1_1 = (
            "Stage 1, Sub-task 1: Apply geometric and algebraic methods to compute initial expressions for r_i and r_o based on stage_0 outputs and former iterations of stage_1.subtask_2. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & former iterations of stage_1.subtask_2, respectively."
        )
        final_decision_instruction_1_1 = (
            "Stage 1, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for expressions of r_i and r_o difference."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["thinking of previous stage_1.subtask_2 iterations"] * 2
        }
        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=3
        )
        logs.append(log_1_1)

        revise_instruction_1_2 = (
            "Stage 1, Sub-task 2: Simplify and refine the expressions for r_i - r_o to a reduced fraction form m/n. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
        )
        revise_desc_1_2 = {
            "instruction": revise_instruction_1_2,
            "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.revise(
            subtask_id="stage_1.subtask_2",
            revise_desc=revise_desc_1_2
        )
        logs.append(log_1_2)

        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1['thinking'])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1['answer'])
        loop_results["stage_1.subtask_2"]["thinking"].append(results_1_2['thinking'])
        loop_results["stage_1.subtask_2"]["answer"].append(results_1_2['answer'])

    aggregate_instruction_2_1 = (
        "Stage 2, Sub-task 1: Evaluate candidate simplified fractions for r_i - r_o and select the one with relatively prime numerator and denominator. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of all stage_1.subtask_2 iterations", "answer of all stage_1.subtask_2 iterations"]
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id="stage_2.subtask_1",
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Stage 3, Sub-task 1: Verify correctness and consistency of the selected fraction and compute the sum m+n as the final answer. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2 & stage_2.subtask_1, respectively."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"] + [results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of all stage_1.subtask_2 iterations", "answer of all stage_1.subtask_2 iterations", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
