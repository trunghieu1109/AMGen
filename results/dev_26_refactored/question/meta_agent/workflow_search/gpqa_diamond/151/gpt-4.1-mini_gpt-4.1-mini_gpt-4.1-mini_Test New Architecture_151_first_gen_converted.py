async def forward_151(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction0_1 = (
        "Sub-task 1: Extract and summarize all relevant biological and experimental information from the query, "
        "including the peptide treatment, shmoo formation, chromatin state, and the protein complexes listed."
    )
    cot_agent_desc0_1 = {
        "instruction": cot_instruction0_1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context": ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    debate_instruction1_1 = (
        "Sub-task 1: Integrate and analyze the biological roles and timing of the four protein complexes "
        "(pre-initiation complex, pre-replication complex, enhancer protein complex, nucleosome histone complex) "
        "in the context of active chromatin during shmoo formation in yeast."
    )
    final_decision_instruction1_1 = (
        "Sub-task 1: Decide on the detailed roles and timing of these complexes in active chromatin during shmoo formation."
    )
    debate_desc1_1 = {
        "instruction": debate_instruction1_1,
        "final_decision_instruction": final_decision_instruction1_1,
        "input": [taskInfo, results0_1["thinking"], results0_1["answer"]],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_instruction1_2 = (
        "Sub-task 2: Analyze the principles and limitations of chromatin immunoprecipitation followed by mass spectrometry (ChIP-MS) "
        "in detecting protein complexes associated with active chromatin."
    )
    cot_agent_desc1_2 = {
        "instruction": cot_instruction1_2,
        "final_decision_instruction": "Sub-task 2: Synthesize and choose the most consistent understanding of ChIP-MS limitations.",
        "input": [taskInfo, results0_1["thinking"], results0_1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    cot_reflect_instruction1_3 = (
        "Sub-task 3: Combine insights from biological roles of complexes and ChIP-MS methodology to evaluate "
        "which protein complex is least likely to be detected in the active chromatin proteome of the shmoo."
    )
    critic_instruction1_3 = (
        "Please review and provide the limitations of provided solutions regarding the detection likelihood "
        "of protein complexes in ChIP-MS assays of active chromatin in shmoo."
    )
    cot_reflect_desc1_3 = {
        "instruction": cot_reflect_instruction1_3,
        "critic_instruction": critic_instruction1_3,
        "input": [
            taskInfo,
            results1_1["thinking"], results1_1["answer"],
            results1_2["thinking"], results1_2["answer"]
        ],
        "temperature": 0.0,
        "context_desc": [
            "user query",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"
        ]
    }
    results1_3, log1_3 = await self.reflexion(
        subtask_id="stage_1.subtask_3",
        reflect_desc=cot_reflect_desc1_3,
        n_repeat=self.max_round
    )
    logs.append(log1_3)

    debate_instruction2_1 = (
        "Sub-task 1: Select the protein complex least likely to be observed in the ChIP-MS assay of active chromatin in the shmoo, "
        "based on integrated biological and methodological analysis."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Finalize the selection of the least likely observed protein complex in the assay."
    )
    debate_desc2_1 = {
        "instruction": debate_instruction2_1,
        "final_decision_instruction": final_decision_instruction2_1,
        "input": [taskInfo, results1_3["thinking"], results1_3["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"],
        "temperature": 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1["thinking"], results2_1["answer"])
    return final_answer, logs
