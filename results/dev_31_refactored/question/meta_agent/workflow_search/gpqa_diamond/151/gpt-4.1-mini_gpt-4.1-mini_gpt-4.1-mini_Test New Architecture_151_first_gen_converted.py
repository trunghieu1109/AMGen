async def forward_151(self, taskInfo):
    logs = []

    results_stage_0 = {}
    results_stage_1 = {}
    results_stage_2 = {}
    results_stage_3 = {"thinking": [], "answer": []}

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the biological effect of the quorum-sensing peptide on yeast and the significance of shmoo formation in relation to active chromatin and gene expression. "
        "Input: taskInfo containing the question and choices."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    results_stage_0["stage_0.subtask_1"] = results_0_1
    logs.append(log_0_1)

    cot_instruction_1_1 = (
        "Sub-task 1: Classify the given protein complexes (pre-initiation, pre-replication, enhancer, nucleosome histone) based on their roles in transcription, replication, and chromatin structure. "
        "Input: results (thinking and answer) from stage_0.subtask_1 and taskInfo."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    results_stage_1["stage_1.subtask_1"] = results_1_1
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1: Identify which protein complexes are expected to be enriched or depleted in active chromatin proteome from ChIP-MS of shmoo cells. "
        "Input: results (thinking and answer) from stage_0.subtask_1 and stage_1.subtask_1, and taskInfo."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"], results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2_1)
    results_stage_2["stage_2.subtask_1"] = results_2_1
    logs.append(log_2_1)

    loop_results_stage_3 = {
        "stage_3.subtask_1": {"thinking": [], "answer": []},
        "stage_3.subtask_2": {"thinking": [], "answer": []}
    }

    for i in range(2):
        cot_instruction_3_1 = (
            f"Iteration {i+1} - Sub-task 1: Generate an initial hypothesis on which complex would be least observed in the active chromatin proteome of shmoo cells. "
            f"Input: taskInfo and all previous thinking and answers from stage_0.subtask_1, stage_1.subtask_1, and stage_2.subtask_1."
        )
        cot_agent_desc_3_1 = {
            "instruction": cot_instruction_3_1,
            "input": [taskInfo] + 
                     [results_0_1["thinking"], results_0_1["answer"],
                      results_1_1["thinking"], results_1_1["answer"],
                      results_2_1["thinking"], results_2_1["answer"]] + 
                     loop_results_stage_3["stage_3.subtask_1"]["thinking"] + 
                     loop_results_stage_3["stage_3.subtask_1"]["answer"] + 
                     loop_results_stage_3["stage_3.subtask_2"]["thinking"] + 
                     loop_results_stage_3["stage_3.subtask_2"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking and answer of previous stages and iterations"]
        }
        results_3_1, log_3_1 = await self.cot(subtask_id=f"stage_3.subtask_1.iter{i+1}", cot_agent_desc=cot_agent_desc_3_1)
        loop_results_stage_3["stage_3.subtask_1"]["thinking"].append(results_3_1["thinking"])
        loop_results_stage_3["stage_3.subtask_1"]["answer"].append(results_3_1["answer"])
        logs.append(log_3_1)

        cot_instruction_3_2 = (
            f"Iteration {i+1} - Sub-task 2: Refine the hypothesis by considering biological context, cell cycle stage, and chromatin activity, using outputs from previous iteration if applicable. "
            f"Input: taskInfo and all thinking and answers from stage_0.subtask_1, stage_1.subtask_1, stage_2.subtask_1, and all previous iterations of stage_3.subtask_1 and stage_3.subtask_2."
        )
        cot_agent_desc_3_2 = {
            "instruction": cot_instruction_3_2,
            "input": [taskInfo] + 
                     [results_0_1["thinking"], results_0_1["answer"],
                      results_1_1["thinking"], results_1_1["answer"],
                      results_2_1["thinking"], results_2_1["answer"]] + 
                     loop_results_stage_3["stage_3.subtask_1"]["thinking"] + 
                     loop_results_stage_3["stage_3.subtask_1"]["answer"] + 
                     loop_results_stage_3["stage_3.subtask_2"]["thinking"] + 
                     loop_results_stage_3["stage_3.subtask_2"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking and answer of previous stages and iterations"]
        }
        results_3_2, log_3_2 = await self.cot(subtask_id=f"stage_3.subtask_2.iter{i+1}", cot_agent_desc=cot_agent_desc_3_2)
        loop_results_stage_3["stage_3.subtask_2"]["thinking"].append(results_3_2["thinking"])
        loop_results_stage_3["stage_3.subtask_2"]["answer"].append(results_3_2["answer"])
        logs.append(log_3_2)

    cot_instruction_4_1 = (
        "Sub-task 1: Evaluate and select the protein complex that best fits the criteria of being least present in the active chromatin proteome of shmoo cells. "
        "Input: taskInfo and all thinking and answers from stage_3.subtask_1 and stage_3.subtask_2 iterations."
    )
    final_decision_instruction_4_1 = (
        "Sub-task 1: Synthesize and choose the best candidate protein complex least observed in the assay based on the debate."
    )
    debate_desc_4_1 = {
        "instruction": cot_instruction_4_1,
        "final_decision_instruction": final_decision_instruction_4_1,
        "input": [taskInfo] + 
                 loop_results_stage_3["stage_3.subtask_1"]["thinking"] + 
                 loop_results_stage_3["stage_3.subtask_1"]["answer"] + 
                 loop_results_stage_3["stage_3.subtask_2"]["thinking"] + 
                 loop_results_stage_3["stage_3.subtask_2"]["answer"],
        "context_desc": ["user query", "thinking and answer of stage_3 iterations"],
        "temperature": 0.5
    }
    results_4_1, log_4_1 = await self.debate(subtask_id="stage_4.subtask_1", debate_desc=debate_desc_4_1, n_repeat=2)
    logs.append(log_4_1)

    aggregate_instruction_4_2 = (
        "Sub-task 2: From solutions generated in Subtask 1, aggregate these solutions and return the consistent and the best solution for selecting the least observed protein complex. "
        "Input: taskInfo and thinking and answer from stage_4.subtask_1."
    )
    aggregate_desc_4_2 = {
        "instruction": aggregate_instruction_4_2,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking and answer of stage_4.subtask_1"]
    }
    results_4_2, log_4_2 = await self.aggregate(subtask_id="stage_4.subtask_2", aggregate_desc=aggregate_desc_4_2)
    logs.append(log_4_2)

    cot_instruction_4_3 = (
        "Sub-task 3: Review and refine the aggregated solution by considering biological context, experimental conditions, and previous reasoning. "
        "Input: taskInfo and thinking and answer from stage_4.subtask_1 and stage_4.subtask_2."
    )
    critic_instruction_4_3 = (
        "Please review and provide the limitations of the provided solutions for selecting the least observed protein complex in the assay."
    )
    cot_reflect_desc_4_3 = {
        "instruction": cot_instruction_4_3,
        "critic_instruction": critic_instruction_4_3,
        "input": [taskInfo, results_4_1["thinking"], results_4_1["answer"], results_4_2["thinking"], results_4_2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking and answer of stage_4.subtask_1", "thinking and answer of stage_4.subtask_2"]
    }
    results_4_3, log_4_3 = await self.reflexion(subtask_id="stage_4.subtask_3", reflect_desc=cot_reflect_desc_4_3, n_repeat=2)
    logs.append(log_4_3)

    cot_agent_instruction_4_4 = (
        "Sub-task 4: Generate the final concise answer selecting the protein complex least observed in the active chromatin proteome of shmoo cells. "
        "Input: taskInfo and thinking and answer from stage_4.subtask_3."
    )
    cot_agent_desc_4_4 = {
        "instruction": cot_agent_instruction_4_4,
        "input": [taskInfo, results_4_3["thinking"], results_4_3["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking and answer of stage_4.subtask_3"]
    }
    results_4_4, log_4_4 = await self.answer_generate(subtask_id="stage_4.subtask_4", cot_agent_desc=cot_agent_desc_4_4)
    logs.append(log_4_4)

    final_answer = await self.make_final_answer(results_4_4["thinking"], results_4_4["answer"])
    return final_answer, logs
