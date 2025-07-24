async def forward_5(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Calculate the volume of tetrahedron ABCD using given edge lengths. "
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

    # stage_0.subtask_2
    cot_instruction_0_2 = (
        "Stage 0, Sub-task 2: Compute the areas of all four faces of the tetrahedron. "
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
        "stage_1.subtask_1": {"thinking": [], "answer": []}
    }

    for i in range(3):
        cot_reflect_instruction_1_1 = (
            f"Stage 1, Sub-task 1, Iteration {i+1}: Formulate the expression for the distance from point I to each face assuming equal distances, "
            "using volume and face areas. Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
        )
        critic_instruction_1_1 = (
            f"Stage 1, Sub-task 1, Iteration {i+1}, Criticism: Review and provide limitations of the formulated distance expressions."
        )
        cot_reflect_desc_1_1 = {
            "instruction": cot_reflect_instruction_1_1,
            "critic_instruction": critic_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.6,
            "context_desc": [
                "user query",
                "thinking of stage_0.subtask_2",
                "answer of stage_0.subtask_2",
            ] + [f"thinking of stage_1.subtask_1 iteration {j+1}" for j in range(len(loop_results["stage_1.subtask_1"]["thinking"]))] + [f"answer of stage_1.subtask_1 iteration {j+1}" for j in range(len(loop_results["stage_1.subtask_1"]["answer"]))]
        }
        results_1_1, log_1_1 = await self.reflexion(
            subtask_id=f"stage_1.subtask_1.iter_{i+1}",
            reflect_desc=cot_reflect_desc_1_1,
            n_repeat=1
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1['thinking'])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1['answer'])

    debate_instruction_2_1 = (
        "Stage 2, Sub-task 1: Assess the candidate distance expressions for correctness and consistency, "
        "selecting the valid simplified form. Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    final_decision_instruction_2_1 = (
        "Stage 2, Sub-task 1, Final Decision: Select the best valid simplified distance expression from candidates generated in stage_1.subtask_1 iterations."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
        "context_desc": ["user query"] + [f"thinking of stage_1.subtask_1 iteration {i+1}" for i in range(3)] + [f"answer of stage_1.subtask_1 iteration {i+1}" for i in range(3)],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=1
    )
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Stage 3, Sub-task 1: Verify the final distance expression meets problem conditions and extract m, n, p to compute m+n+p. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
