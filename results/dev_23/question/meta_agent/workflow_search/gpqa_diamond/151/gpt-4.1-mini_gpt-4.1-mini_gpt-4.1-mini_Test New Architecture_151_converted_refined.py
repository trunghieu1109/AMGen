async def forward_151(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the biological context of the experiment, including the quorum-sensing peptide treatment, "
        "shmoo formation in Saccharomyces cerevisiae, and implications for active chromatin and gene expression. "
        "Embed feedback by emphasizing the need to clearly define 'active chromatin' in this context and consider the cell cycle stage during shmoo formation to avoid misinterpretation of complex activity."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze and classify the four protein complexes (pre-initiation complex, pre-replication complex, "
        "enhancer protein complex, nucleosome histone complex) with a focus on their known existence and roles specifically in Saccharomyces cerevisiae. "
        "Explicitly verify organism-specific presence or absence of each complex to avoid misapplication of metazoan concepts (e.g., classical enhancer complexes) to yeast. "
        "This step addresses previous errors caused by assuming enhancer complexes exist in yeast."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Analyze the chromatin immunoprecipitation followed by mass spectrometry (ChIP-MS) technique to understand which types of protein complexes are typically recovered from active chromatin regions. "
        "Emphasize the assay's specificity and limitations, particularly regarding detection likelihood of transcription-related versus replication-related complexes, "
        "to prevent inconsistent conclusions in later subtasks."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Integrate and reconcile insights from subtasks 1, 2, and 3 to evaluate and compare the likelihood of detecting each protein complex in the active chromatin of the shmoo using ChIP-MS. "
        "Explicitly check for consistency and biological plausibility, flag contradictions, and avoid previous errors where transcription-related complexes were incorrectly ranked as least observed. "
        "This subtask must produce a coherent, consistent ranking of complexes based on integrated evidence."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
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

    reflexion_instruction5 = (
        "Sub-task 5: Select the protein complex that would be least observed in the ChIP-MS assay of active chromatin in the shmoo and justify the choice based on the fully integrated and reconciled analysis from subtask 4. "
        "This subtask must explicitly reference and incorporate conclusions from all previous subtasks to ensure the final answer is consistent, biologically sound, and aligned with assay characteristics. "
        "Include a reflexive review step to confirm no contradictions remain."
    )
    reflexion_desc5 = {
        'instruction': reflexion_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=reflexion_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
