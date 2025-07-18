async def forward_188(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each effective particle (Magnon, Skyrmion, Pion, Phonon) "
        "based on whether it is associated with a spontaneously-broken symmetry, considering the nature of the symmetry and the physical context. "
        "Provide detailed reasoning for each particle, highlighting the connection or lack thereof to spontaneous symmetry breaking."
    )
    debate_desc1 = {
        'instruction': cot_instruction1,
        'final_decision_instruction': (
            "Sub-task 1: After debate, provide a clear classification for each particle regarding spontaneous symmetry breaking association."
        ),
        'input': [taskInfo],
        'context_desc': ["user query"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the debate classification results from Sub-task 1, evaluate and identify which particle is not associated with spontaneous symmetry breaking. "
        "Justify the selection with clear physical principles and definitions of spontaneous symmetry breaking and effective particles."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct solution for which particle is not associated with spontaneous symmetry breaking, "
        "given the debate outputs."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
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

    final_answer = await self.make_final_answer(results2['thinking'], results2['answer'])
    return final_answer, logs
