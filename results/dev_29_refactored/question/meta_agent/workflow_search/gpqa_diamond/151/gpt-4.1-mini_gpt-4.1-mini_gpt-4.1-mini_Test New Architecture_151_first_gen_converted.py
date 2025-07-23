async def forward_151(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": []}

    for iteration in range(3):
        cot_instruction_0 = (
            "Sub-task 0: Extract and summarize the given information from the query to identify key experimental details and question focus."
        )
        cot_agent_desc_0 = {
            "instruction": cot_instruction_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0, log_0 = await self.cot(
            subtask_id=f"stage0_subtask0_iter{iteration}",
            cot_agent_desc=cot_agent_desc_0
        )
        logs.append(log_0)
        stage0_results["subtask_0"].append(results_0)

        cot_instruction_1 = (
            "Sub-task 1: Analyze relationships between biological components and complexes mentioned to understand their roles in the assay context, "
            "based on the summary from Sub-task 0."
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_1, log_1 = await self.cot(
            subtask_id=f"stage0_subtask1_iter{iteration}",
            cot_agent_desc=cot_agent_desc_1
        )
        logs.append(log_1)
        stage0_results["subtask_1"].append(results_1)

        cot_instruction_2 = (
            "Sub-task 2: Identify the relevant scientific fields and disciplines involved to frame the problem correctly, "
            "based on the analysis from Sub-task 1."
        )
        cot_agent_desc_2 = {
            "instruction": cot_instruction_2,
            "input": [taskInfo, results_1["thinking"], results_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_2, log_2 = await self.cot(
            subtask_id=f"stage0_subtask2_iter{iteration}",
            cot_agent_desc=cot_agent_desc_2
        )
        logs.append(log_2)
        stage0_results["subtask_2"].append(results_2)

        cot_instruction_3 = (
            "Sub-task 3: Highlight aspects of the query that need clarification or assumptions to be made for reasoning, "
            "based on the fields identified in Sub-task 2."
        )
        cot_agent_desc_3 = {
            "instruction": cot_instruction_3,
            "input": [taskInfo, results_2["thinking"], results_2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results_3, log_3 = await self.cot(
            subtask_id=f"stage0_subtask3_iter{iteration}",
            cot_agent_desc=cot_agent_desc_3
        )
        logs.append(log_3)
        stage0_results["subtask_3"].append(results_3)

    last_subtask3_thinking = [r["thinking"] for r in stage0_results["subtask_3"]]
    last_subtask3_answer = [r["answer"] for r in stage0_results["subtask_3"]]

    reflexion_instruction_0 = (
        "Sub-task 0: Simplify and consolidate the extracted information and analysis to focus on the key criteria for selecting the least observed protein complex, "
        "based on the clarifications from Stage 0 Sub-task 3 iterations."
    )
    reflexion_desc_0 = {
        "instruction": reflexion_instruction_0,
        "input": [taskInfo] + last_subtask3_thinking + last_subtask3_answer,
        "temperature": 0.0,
        "context_desc": ["user query"] + ["thinking of stage0_subtask3"]*3 + ["answer of stage0_subtask3"]*3
    }
    results_reflexion_0, log_reflexion_0 = await self.reflexion(
        subtask_id="stage1_subtask0",
        reflect_desc=reflexion_desc_0,
        n_repeat=self.max_round
    )
    logs.append(log_reflexion_0)

    answergen_instruction_1 = (
        "Sub-task 1: Evaluate each candidate protein complex against the criteria derived from the biological context and assay method to select the least likely observed complex, "
        "based on the simplified and consolidated information from Sub-task 0."
    )
    answergen_desc_1 = {
        "instruction": answergen_instruction_1,
        "input": [taskInfo, results_reflexion_0["thinking"], results_reflexion_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage1_subtask0", "answer of stage1_subtask0"]
    }
    results_answergen_1, log_answergen_1 = await self.answer_generate(
        subtask_id="stage1_subtask1",
        cot_agent_desc=answergen_desc_1
    )
    logs.append(log_answergen_1)

    cot_instruction_2 = (
        "Sub-task 0: Apply biological knowledge and reasoning to transform the refined candidate selection into a definitive answer regarding the least observed complex, "
        "based on the evaluation from Stage 1 Sub-task 1."
    )
    cot_agent_desc_2 = {
        "instruction": cot_instruction_2,
        "input": [taskInfo, results_answergen_1["thinking"], results_answergen_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results_stage2_0, log_stage2_0 = await self.cot(
        subtask_id="stage2_subtask0",
        cot_agent_desc=cot_agent_desc_2
    )
    logs.append(log_stage2_0)

    answergen_instruction_2 = (
        "Sub-task 0: Generate the definitive answer for the least observed protein complex, "
        "based on the reasoning from the previous CoT subtask in Stage 2."
    )
    answergen_desc_2 = {
        "instruction": answergen_instruction_2,
        "input": [taskInfo, results_stage2_0["thinking"], results_stage2_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage2_subtask0", "answer of stage2_subtask0"]
    }
    results_stage2_1, log_stage2_1 = await self.answer_generate(
        subtask_id="stage2_subtask1",
        cot_agent_desc=answergen_desc_2
    )
    logs.append(log_stage2_1)

    reflexion_instruction_3 = (
        "Sub-task 0: Validate the selected answer for correctness, consistency with biological principles, and alignment with the experimental context, "
        "based on the definitive answer from Stage 2."
    )
    critic_instruction_3 = (
        "Please review and provide the limitations of the provided solution regarding the least observed protein complex in the assay context."
    )
    reflexion_desc_3 = {
        "instruction": reflexion_instruction_3,
        "critic_instruction": critic_instruction_3,
        "input": [taskInfo, results_stage2_1["thinking"], results_stage2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage2_subtask1", "answer of stage2_subtask1"]
    }
    results_reflexion_3, log_reflexion_3 = await self.reflexion(
        subtask_id="stage3_subtask0",
        reflect_desc=reflexion_desc_3,
        n_repeat=self.max_round
    )
    logs.append(log_reflexion_3)

    final_answer = await self.make_final_answer(results_stage2_1["thinking"], results_stage2_1["answer"])

    return final_answer, logs
