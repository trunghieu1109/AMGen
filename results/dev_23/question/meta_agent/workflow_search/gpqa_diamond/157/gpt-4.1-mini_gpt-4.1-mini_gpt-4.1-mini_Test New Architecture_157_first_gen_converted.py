async def forward_157(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize the key biological components and mutation characteristics from the query, including protein domains, mutation types, and their genetic dominance/recessiveness."
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

    cot_sc_instruction2 = "Sub-task 2: Classify the functional roles of the transcription factor domains (transactivation and dimerization) and the typical molecular consequences of mutations in these domains, especially dominant-negative mutations."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Analyze the molecular mechanisms by which a dominant-negative mutation in the dimerization domain can affect protein behavior and function, considering options like loss of dimerization, aggregation, degradation, or conformational changes."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Evaluate each provided molecular phenotype option against the known effects of dominant-negative mutations in dimerization domains to identify the most plausible phenotype caused by mutation Y."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = "Sub-task 5: Synthesize the analysis to select the most likely molecular phenotype observed in the presence of mutation Y and justify the choice based on the prior subtasks."
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
