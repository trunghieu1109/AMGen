async def forward_151(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the biological context of the experiment involving quorum-sensing peptide treatment, shmoo formation in Saccharomyces cerevisiae, and implications for active chromatin and gene expression with context from taskInfo"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = "Sub-task 2: Analyze and classify the four protein complexes (pre-initiation complex, pre-replication complex, enhancer protein complex, nucleosome histone complex) in terms of their roles in chromatin biology and transcriptional activity with context from taskInfo"
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Analyze the chromatin immunoprecipitation followed by mass spectrometry (ChIP-MS) technique to understand what types of protein complexes are typically recovered from active chromatin regions with context from taskInfo"
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Evaluate and compare the likelihood of detecting each protein complex in the active chromatin of the shmoo using ChIP-MS, integrating insights from biological context, complex functions, and assay characteristics"
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Select the protein complex that would be least observed in the ChIP-MS assay of active chromatin in the shmoo and justify the choice based on the integrated analysis"
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
