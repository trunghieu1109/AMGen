async def forward_151(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Analyze the biological context of shmoo formation in Saccharomyces cerevisiae induced by the quorum-sensing peptide, focusing on the chromatin state and gene expression activity."
    cot_sc_final_decision_instruction1 = "Sub-task 1: Synthesize and choose the most consistent understanding of the biological context of shmoo formation and chromatin activity."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': cot_sc_final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, classify and describe the functions and typical chromatin association of each protein complex listed: pre-initiation complex, pre-replication complex, enhancer protein complex, and nucleosome histone complex."
    cot_sc_final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent classification and description of the protein complexes with respect to chromatin association."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': cot_sc_final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Evaluate which of the four protein complexes (pre-initiation complex, pre-replication complex, enhancer protein complex, nucleosome histone complex) is least likely to be enriched or detected by chromatin immunoprecipitation from active chromatin during shmoo formation, based on their biological roles and chromatin association."
    debate_final_decision_instruction3 = "Sub-task 3: Decide which protein complex is least observed in the assay based on the debate."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': debate_final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = "Sub-task 4: Synthesize the evaluation results from the debate to select the protein complex least observed in the assay and justify the choice with molecular biology reasoning."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
