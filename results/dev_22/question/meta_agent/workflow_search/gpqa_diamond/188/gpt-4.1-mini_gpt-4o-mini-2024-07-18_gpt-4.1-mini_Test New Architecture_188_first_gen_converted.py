async def forward_188(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze and classify each effective particle (Magnon, Skyrmion, Pion, Phonon) "
        "in terms of their physical origin and whether they arise from spontaneous symmetry breaking, "
        "with context from the query about effective particles and spontaneous symmetry breaking."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "task decomposition", "detailed analysis"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Clarify the nature of the symmetries involved (continuous vs discrete, global vs local) "
        "and the type of spontaneous symmetry breaking relevant to each particle to refine the classification, "
        "based on the analysis from Sub-task 1."
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

    debate_instruction3 = (
        "Sub-task 3: Evaluate and compare the classifications of the four particles to identify which one is not associated "
        "with spontaneous symmetry breaking, based on outputs from Sub-task 1 and Sub-task 2."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Synthesize the findings into a clear, justified conclusion answering which particle is not linked "
        "to spontaneous symmetry breaking, based on the evaluation from Sub-task 3."
    )
    cot_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_desc4,
        n_repeat=1
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
