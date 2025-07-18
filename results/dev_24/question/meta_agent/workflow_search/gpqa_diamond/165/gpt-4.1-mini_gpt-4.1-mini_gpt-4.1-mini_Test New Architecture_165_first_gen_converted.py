async def forward_165(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the given Lagrangian, field content, and vacuum expectation values "
        "to understand the model setup and the role of each field, with context from taskInfo"
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, identify and clarify the physical meaning of the pseudo-Goldstone boson H2 "
        "and the mechanism of its mass generation via radiative corrections in the context of the model."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent understanding of the pseudo-Goldstone boson H2 and its mass generation mechanism, "
        "given all the above thinking and answers."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
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

    debate_instruction3 = (
        "Sub-task 3: Select and evaluate the candidate mass formulae for the pseudo-Goldstone boson H2, "
        "focusing on the structure of terms, signs, and dependence on VEVs and loop factors, with context from previous subtasks."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Compare the candidate formulae against theoretical expectations for radiative corrections, "
        "including contributions from bosons and fermions, and the correct scaling with VEVs, to identify the most accurate approximation."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Synthesize the analysis to select the best approximation formula for the pseudo-Goldstone boson mass squared "
        "and justify the choice based on the model and radiative correction principles, using outputs from the debate."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
