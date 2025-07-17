async def forward_151(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the given biological elements: quorum-sensing peptide, shmoo formation, active chromatin, "
        "and listed protein complexes (pre-initiation complex, pre-replication complex, enhancer protein complex, nucleosome histone complex), "
        "focusing on their functions and relevance to transcriptional activity in yeast, with context from taskInfo."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, debate which protein complexes among pre-initiation complex, pre-replication complex, "
        "enhancer protein complex, and nucleosome histone complex are associated with active chromatin during shmoo formation in yeast, "
        "and which are least likely to be detected by chromatin immunoprecipitation followed by mass spectrometry, with context from taskInfo and Sub-task 1 outputs."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, transform the evaluation into a clear conclusion identifying "
        "the protein complex least represented in the active chromatin proteome under the experimental conditions, with context from taskInfo and previous subtasks."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
