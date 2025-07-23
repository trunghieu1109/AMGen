async def forward_151(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the biological effect of the quorum-sensing peptide on yeast, focusing on shmoo formation and its relation to active chromatin and gene expression changes. "
        "Summarize key experimental details relevant to chromatin immunoprecipitation and mass spectrometry assay context. "
        "Input: taskInfo containing the user query and choices."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_1_1 = (
        "Sub-task 1: Classify the given protein complexes (pre-initiation, pre-replication, enhancer, nucleosome histone) based on their canonical roles in transcription, replication, and chromatin structure. "
        "Explicitly note their general association with active or inactive chromatin states, without yet considering local chromatin dynamics or assay-specific context. "
        "Input: results (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1: Critically evaluate each protein complex's expected occupancy in the active chromatin proteome of shmoo cells, explicitly incorporating knowledge of local nucleosome eviction at active promoters and the specific chromatin immunoprecipitation target (e.g., active marks or RNA polymerase II). "
        "Address the previous failure of assuming nucleosomes are always present and enriched, and determine which complexes are likely enriched or depleted in the assay. "
        "Input: results (thinking and answer) from stage_0.subtask_1 and stage_1.subtask_1."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    loop_results = {
        "stage_3.subtask_1": {"thinking": [], "answer": []},
        "stage_3.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_3_1 = (
            "Sub-task 1: Generate an initial hypothesis on which protein complex would be least observed in the active chromatin proteome of shmoo cells based on validated occupancy from stage_2. "
            "Explicitly consider biological context, cell cycle stage, and chromatin activity to avoid previous errors. "
            "Input: results (thinking and answer) from stage_2.subtask_1."
        )
        cot_agent_desc_3_1 = {
            "instruction": cot_instruction_3_1,
            "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + [results_2_1['thinking'], results_2_1['answer']],
            "temperature": 0.0,
            "context": ["user query"] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + ["thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
        }
        results_3_1, log_3_1 = await self.cot(
            subtask_id=f"stage_3.subtask_1.iter{iteration+1}",
            cot_agent_desc=cot_agent_desc_3_1
        )
        logs.append(log_3_1)
        loop_results["stage_3.subtask_1"]["thinking"].append(results_3_1["thinking"])
        loop_results["stage_3.subtask_1"]["answer"].append(results_3_1["answer"])

        cot_reflect_instruction_3_2 = (
            "Sub-task 2: Refine the initial hypothesis by critically challenging assumptions and integrating additional biological insights or potential experimental caveats. "
            "Use outputs from the initial hypothesis and prior stages to ensure robustness and avoid overlooking key factors such as nucleosome eviction. "
            "Input: results (thinking and answer) from stage_3.subtask_1 and stage_2.subtask_1."
        )
        critic_instruction_3_2 = (
            "Please review and provide the limitations of provided solutions of the initial hypothesis on least observed protein complex in active chromatin proteome of shmoo cells."
        )
        cot_reflect_desc_3_2 = {
            "instruction": cot_reflect_instruction_3_2,
            "critic_instruction": critic_instruction_3_2,
            "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + [results_2_1['thinking'], results_2_1['answer']],
            "temperature": 0.0,
            "context": ["user query"] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + ["thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
        }
        results_3_2, log_3_2 = await self.reflexion(
            subtask_id=f"stage_3.subtask_2.iter{iteration+1}",
            reflect_desc=cot_reflect_desc_3_2,
            n_repeat=self.max_round
        )
        logs.append(log_3_2)
        loop_results["stage_3.subtask_2"]["thinking"].append(results_3_2["thinking"])
        loop_results["stage_3.subtask_2"]["answer"].append(results_3_2["answer"])

    debate_instruction_4_1 = (
        "Sub-task 1: Evaluate and select the protein complex that best fits the criteria of being least present in the active chromatin proteome of shmoo cells, synthesizing reasoning from both initial and refined hypotheses. "
        "Use collaborative methods (CoT, Debate, Aggregate) to ensure a robust and well-justified final answer. "
        "Input: results (thinking and answer) from stage_3.subtask_1 and stage_3.subtask_2."
    )
    final_decision_instruction_4_1 = (
        "Sub-task 1: Select the least observed protein complex in the active chromatin proteome of shmoo cells based on all prior analyses."
    )
    debate_desc_4_1 = {
        "instruction": debate_instruction_4_1,
        "final_decision_instruction": final_decision_instruction_4_1,
        "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + loop_results["stage_3.subtask_2"]["thinking"] + loop_results["stage_3.subtask_2"]["answer"],
        "context_desc": ["user query"] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + loop_results["stage_3.subtask_2"]["thinking"] + loop_results["stage_3.subtask_2"]["answer"],
        "temperature": 0.5
    }
    results_4_1, log_4_1 = await self.debate(
        subtask_id="stage_4.subtask_1",
        debate_desc=debate_desc_4_1,
        n_repeat=self.max_round
    )
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1['thinking'], results_4_1['answer'])
    return final_answer, logs
