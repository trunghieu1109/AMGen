async def forward_165(self, taskInfo):
    logs = []
    stage0_results = {}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize the given Lagrangian, field content, and vacuum expectation values "
            "relevant to the pseudo-Goldstone boson mass problem."
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

        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the structure of radiative corrections contributing to the pseudo-Goldstone boson mass, "
            "identifying relevant particle mass terms and coefficients, based on output from Sub-task 0."
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

        cot_instruction_0_2 = (
            "Sub-task 2: Document the relationships between the VEVs, particle masses, and coefficients in the candidate mass formulas, "
            "noting differences among choices, based on output from Sub-task 1."
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

        stage0_results[iteration] = {
            "subtask_0": results_0_0,
            "subtask_1": results_0_1,
            "subtask_2": results_0_2
        }

    last_iter = 2
    stage0_subtask_2_thinking = stage0_results[last_iter]["subtask_2"]["thinking"]
    stage0_subtask_2_answer = stage0_results[last_iter]["subtask_2"]["answer"]

    cot_reflect_instruction_1_0 = (
        "Sub-task 0: Simplify and consolidate the intermediate analysis to clarify the correct structure of the mass formula, "
        "focusing on the placement of the factor (x^2 + v^2) and the inclusion of fermionic and bosonic contributions."
    )
    critic_instruction_1_0 = (
        "Please review and provide the limitations of provided solutions of the mass formula structure and contributions."
    )
    cot_reflect_desc_1_0 = {
        "instruction": cot_reflect_instruction_1_0,
        "critic_instruction": critic_instruction_1_0,
        "input": [taskInfo, stage0_subtask_2_thinking, stage0_subtask_2_answer],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_0, log_1_0 = await self.reflexion(
        subtask_id="stage_1_subtask_0",
        reflect_desc=cot_reflect_desc_1_0,
        n_repeat=self.max_round
    )
    logs.append(log_1_0)

    debate_instruction_1_1 = (
        "Sub-task 1: Evaluate each candidate formula against theoretical expectations for radiative corrections to pseudo-Goldstone boson masses, "
        "including sign conventions and particle content, based on previous analysis."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Evaluate and debate the correctness of candidate formulas for the pseudo-Goldstone boson mass."
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

    aggregate_instruction_1_2 = (
        "Sub-task 2: Select the best candidate formula that matches the expected theoretical structure and physical reasoning, "
        "aggregating results from reflexion and debate subtasks."
    )
    aggregate_desc_1_2 = {
        "instruction": aggregate_instruction_1_2,
        "input": [taskInfo, results_1_0, results_1_1],
        "temperature": 0.0,
        "context": ["user query", "results from stage_1.subtask_0", "results from stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.aggregate(
        subtask_id="stage_1_subtask_2",
        aggregate_desc=aggregate_desc_1_2
    )
    logs.append(log_1_2)

    cot_agent_instruction_2_0 = (
        "Sub-task 0: Apply the selected formula to the given parameters and interpret the physical meaning of each term "
        "in the context of the extended Standard Model."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_agent_instruction_2_0,
        "input": [taskInfo, results_1_2],
        "temperature": 0.0,
        "context": ["user query", "selected formula from stage_1.subtask_2"]
    }
    results_2_0, log_2_0 = await self.answer_generate(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    review_instruction_3_0 = (
        "Sub-task 0: Validate the selected mass formula for consistency with known theoretical principles such as symmetry breaking, "
        "loop corrections, and sign conventions, based on the applied formula and interpretation."
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
