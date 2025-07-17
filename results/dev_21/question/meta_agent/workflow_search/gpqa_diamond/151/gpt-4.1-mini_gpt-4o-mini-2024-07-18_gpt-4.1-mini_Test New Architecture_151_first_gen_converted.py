async def forward_151(self, taskInfo):
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Analyze the biological context of shmoo formation in Saccharomyces cerevisiae induced by the quorum-sensing peptide, "
        "focusing on the chromatin state and gene expression changes involved, with context from taskInfo."
    )
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_1, log_1 = await self.cot(
        subtask_id='subtask_1',
        cot_agent_desc=cot_agent_desc_1
    )
    logs.append(log_1)

    cot_instruction_2 = (
        "Sub-task 2: Classify and describe the functions and typical chromatin association of each protein complex option: "
        "pre-initiation complex, pre-replication complex, enhancer protein complex, and nucleosome histone complex, with context from taskInfo."
    )
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_2, log_2 = await self.cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_agent_desc_2
    )
    logs.append(log_2)

    debate_instruction_3 = (
        "Sub-task 3: Evaluate the relevance and expected abundance of each protein complex in the active chromatin proteome of the shmoo, "
        "based on their roles in transcription, replication, and chromatin structure, using outputs from subtask_1 and subtask_2."
    )
    debate_desc_3 = {
        'instruction': debate_instruction_3,
        'context': ['user query', results_1['thinking'], results_1['answer'], results_2['thinking'], results_2['answer']],
        'input': [taskInfo, results_1, results_2],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results_3, log_3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc_3,
        n_repeat=self.max_round
    )
    logs.append(log_3)

    debate_instruction_4 = (
        "Sub-task 4: Determine which protein complex would be least represented in the chromatin immunoprecipitation-mass spectrometry assay targeting active chromatin in the shmoo, "
        "based on the evaluation from subtask_3."
    )
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'context': ['user query', results_3['thinking'], results_3['answer']],
        'input': [taskInfo, results_3],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    results_4, log_4 = await self.debate(
        subtask_id='subtask_4',
        debate_desc=debate_desc_4,
        n_repeat=self.max_round
    )
    logs.append(log_4)

    final_answer = await self.make_final_answer(results_4['thinking'], results_4['answer'])
    return final_answer, logs
