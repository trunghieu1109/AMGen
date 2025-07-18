async def forward_151(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction0 = (
        "Sub-task 1: Extract and summarize the key biological and experimental information from the query, "
        "including the peptide treatment, shmoo formation, chromatin immunoprecipitation, and the list of candidate protein complexes."
    )
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent summary of the key biological and experimental information.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    debate_instruction1_1 = (
        "Sub-task 1: Analyze the biological roles and relevance of each protein complex (pre-initiation complex, pre-replication complex, "
        "enhancer protein complex, nucleosome histone complex) in the context of active chromatin during shmoo formation in yeast."
    )
    debate_desc1_1 = {
        'instruction': debate_instruction1_1,
        'final_decision_instruction': "Sub-task 1: Decide the roles and relevance of each protein complex in active chromatin during shmoo formation.",
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        'temperature': 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_reflect_instruction1_2 = (
        "Sub-task 2: Integrate knowledge about chromatin states, transcriptional activity, and cell cycle phases to assess which complexes "
        "are typically present or absent in active chromatin under the experimental conditions of shmoo formation."
    )
    critic_instruction1_2 = (
        "Please review and provide the limitations of the provided analyses regarding the presence or absence of protein complexes in active chromatin during shmoo formation."
    )
    cot_reflect_desc1_2 = {
        'instruction': cot_reflect_instruction1_2,
        'critic_instruction': critic_instruction1_2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1_1['thinking'], results1_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results1_2, log1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Select the protein complex that would be least represented in the active chromatin proteome recovered by chromatin immunoprecipitation "
        "followed by mass spectrometry in shmoo-forming yeast treated with the quorum-sensing peptide."
    )
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'final_decision_instruction': "Sub-task 1: Choose the least represented protein complex in the active chromatin proteome under the given experimental conditions.",
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])
    return final_answer, logs
