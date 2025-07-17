async def forward_188(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each given particle (Magnon, Skyrmion, Pion, Phonon) "
        "based on whether it arises from spontaneous symmetry breaking or not, with detailed reasoning."
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
        "Sub-task 2: Evaluate and compare the classifications from Sub-task 1 to identify which particle "
        "is not associated with spontaneous symmetry breaking, debating the reasoning and evidence."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", results1['thinking'], results1['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Select the final answer based on the evaluation from Sub-task 2 and provide a clear explanation "
        "supporting why the chosen particle is not linked to spontaneous symmetry breaking."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer']]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
