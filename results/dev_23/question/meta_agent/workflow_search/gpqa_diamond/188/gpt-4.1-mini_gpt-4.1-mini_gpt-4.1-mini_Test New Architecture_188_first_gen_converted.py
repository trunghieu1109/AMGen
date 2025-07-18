async def forward_188(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each effective particle (Magnon, Skyrmion, Pion, Phonon) "
        "based on whether it is associated with a spontaneously-broken symmetry, including identifying the type of symmetry broken "
        "and the nature of the particle (Goldstone boson, topological excitation, vibrational mode, etc.)."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the classification from Sub-task 1, evaluate which particle is not associated with spontaneous symmetry breaking, "
        "considering the physical context and definitions of spontaneous symmetry breaking and effective particles."
    )
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

    debate_instruction_3 = (
        "Sub-task 3: Select and justify the final answer identifying the particle not associated with a spontaneously-broken symmetry, "
        "providing clear reasoning and referencing the analysis from previous subtasks."
    )
    debate_desc3 = {
        'instruction': debate_instruction_3,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
