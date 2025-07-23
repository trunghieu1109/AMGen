async def forward_171(self, taskInfo):
    logs = []
    stage_0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": []}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize the given information from the query, including excitation ratio, energy difference, and LTE assumption."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        stage_0_results["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the relationship between excitation ratio and temperatures using the Boltzmann distribution and express the excitation ratio in terms of T1, T2, ΔE, and k."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        stage_0_results["subtask_1"].append(results_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Derive an intermediate equation relating ln(2) to T1 and T2 based on the Boltzmann formula and the given energy difference."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        stage_0_results["subtask_2"].append(results_0_2)

    cot_reflect_instruction_1_0 = (
        "Sub-task 0: Simplify and consolidate the intermediate equation to a form comparable with the candidate equations provided."
    )
    critic_instruction_1_0 = (
        "Please review and provide the limitations of the simplified intermediate equation and its comparability with candidate equations."
    )
    cot_reflect_desc_1_0 = {
        "instruction": cot_reflect_instruction_1_0,
        "critic_instruction": critic_instruction_1_0,
        "input": [taskInfo] + [res["thinking"] for res in stage_0_results["subtask_2"]] + [res["answer"] for res in stage_0_results["subtask_2"]],
        "temperature": 0.0,
        "context": ["user query"] + ["thinking of stage_0.subtask_2"]*3 + ["answer of stage_0.subtask_2"]*3
    }
    results_1_0, log_1_0 = await self.reflexion(
        subtask_id="stage_1_subtask_0",
        reflect_desc=cot_reflect_desc_1_0,
        n_repeat=self.max_round
    )
    logs.append(log_1_0)

    debate_instruction_1_1 = (
        "Sub-task 1: Evaluate each candidate equation against the derived intermediate equation to identify which matches the correct relationship."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Select the candidate equation that best matches the derived intermediate equation and physical reasoning."
    )
    debate_desc_1_1 = {
        "instruction": debate_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_1_0["thinking"], results_1_0["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"],
        "temperature": 0.5
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id="stage_1_subtask_1",
        debate_desc=debate_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_agent_instruction_1_2 = (
        "Sub-task 2: Select the best candidate equation that correctly represents the relationship between T1 and T2 given the excitation ratio and energy difference."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_agent_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.answer_generate(
        subtask_id="stage_1_subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_0 = (
        "Sub-task 0: Apply algebraic transformations or substitutions to the selected candidate equation to verify its consistency with the physical parameters and assumptions."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    review_instruction_3_0 = (
        "Sub-task 0: Validate the final selected equation for correctness, consistency with LTE assumptions, and physical plausibility."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results_2_0["thinking"], results_2_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"]
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id="stage_3_subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    final_answer = await self.make_final_answer(results_3_0["thinking"], results_3_0["answer"])
    return final_answer, logs
