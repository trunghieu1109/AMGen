async def forward_156(self, taskInfo):
    logs = []
    loop_results_stage0 = {"subtask_0": [], "subtask_1": [], "subtask_2": []}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Analyze the viral outbreak context and identify possible viral identification methods "
            "(DNA sequencing, cDNA sequencing, antibody detection, symptom-based inference) with context from the query."
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
        loop_results_stage0["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Generate candidate molecular diagnostic approaches based on identified viral detection methods "
            "(PCR, nested PCR, real-time PCR, ELISA) with context from previous analysis."
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
        loop_results_stage0["subtask_1"].append(results_0_1)

        cot_agent_instruction_0_2 = (
            "Sub-task 2: Document advantages and limitations of each candidate approach with respect to speed, accuracy, and retrovirus biology, "
            "based on previous candidate approaches generated."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_agent_instruction_0_2,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_2, log_0_2 = await self.answer_generate(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        loop_results_stage0["subtask_2"].append(results_0_2)

    # Aggregate all iterations outputs from stage_0.subtask_2 for next stage
    all_advantages_limitations = [res["answer"] for res in loop_results_stage0["subtask_2"]]

    cot_reflect_instruction_1_0 = (
        "Sub-task 0: Simplify and consolidate the candidate diagnostic approaches into a refined set of feasible options, "
        "based on all documented advantages and limitations from stage 0 iterations."
    )
    critic_instruction_1_0 = (
        "Please review and provide the limitations of the provided solutions for consolidating candidate diagnostic approaches."
    )
    cot_reflect_desc_1_0 = {
        "instruction": cot_reflect_instruction_1_0,
        "critic_instruction": critic_instruction_1_0,
        "input": [taskInfo] + all_advantages_limitations,
        "temperature": 0.0,
        "context": ["user query"] + [f"advantage_limitation_iter_{i}" for i in range(3)]
    }
    results_1_0, log_1_0 = await self.reflexion(
        subtask_id="stage_1_subtask_0",
        reflect_desc=cot_reflect_desc_1_0,
        n_repeat=self.max_round
    )
    logs.append(log_1_0)

    debate_instruction_1_1 = (
        "Sub-task 1: Evaluate refined candidate diagnostic approaches against criteria of quickness, accuracy, "
        "and suitability for retrovirus detection, debating their merits and drawbacks."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Select the best candidate diagnostic approach based on debate evaluation."
    )
    debate_desc_1_1 = {
        "instruction": debate_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_1_0["thinking"], results_1_0["answer"]],
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "temperature": 0.5
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id="stage_1_subtask_1",
        debate_desc=debate_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    aggregate_instruction_1_2 = (
        "Sub-task 2: From solutions generated in Subtask 1, aggregate these solutions and return the consistent and the best solution for selecting the diagnostic approach."
    )
    aggregate_desc_1_2 = {
        "instruction": aggregate_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from subtask 1"]
    }
    results_1_2, log_1_2 = await self.aggregate(
        subtask_id="stage_1_subtask_2",
        aggregate_desc=aggregate_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_0 = (
        "Sub-task 0: Develop a detailed design of the molecular diagnostic kit based on the selected candidate approach, "
        "including primer design for PCR or antibody selection for ELISA, with context from the selected best candidate."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage 1 subtask 2", "answer of stage 1 subtask 2"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    cot_agent_instruction_2_1 = (
        "Sub-task 1: Outline the workflow and protocol for the diagnostic kit to ensure quick and accurate detection, "
        "based on the detailed design developed."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_agent_instruction_2_1,
        "input": [taskInfo, results_2_0["thinking"], results_2_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage 2 subtask 0", "answer of stage 2 subtask 0"]
    }
    results_2_1, log_2_1 = await self.answer_generate(
        subtask_id="stage_2_subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    review_instruction_3_0 = (
        "Sub-task 0: Evaluate the designed diagnostic kit against predefined criteria: speed of detection, accuracy, specificity to retrovirus, and feasibility."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage 2 subtask 1", "answer of stage 2 subtask 1"]
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id="stage_3_subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    cot_reflect_instruction_3_1 = (
        "Sub-task 1: Provide feedback and suggest improvements or confirm readiness of the diagnostic kit design, "
        "based on the evaluation results."
    )
    critic_instruction_3_1 = (
        "Please review and provide constructive feedback or confirm the design readiness for the diagnostic kit."
    )
    cot_reflect_desc_3_1 = {
        "instruction": cot_reflect_instruction_3_1,
        "critic_instruction": critic_instruction_3_1,
        "input": [taskInfo, results_3_0["thinking"], results_3_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage 3 subtask 0", "answer of stage 3 subtask 0"]
    }
    results_3_1, log_3_1 = await self.reflexion(
        subtask_id="stage_3_subtask_1",
        reflect_desc=cot_reflect_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
