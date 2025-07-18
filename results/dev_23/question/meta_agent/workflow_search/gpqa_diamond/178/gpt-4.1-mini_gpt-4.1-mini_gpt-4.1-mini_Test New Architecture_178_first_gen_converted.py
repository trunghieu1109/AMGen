async def forward_178(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the given matrices W, X, Y, and Z to determine their key properties relevant to quantum mechanics: "
        "check if W and X are unitary (evolution operators), if Y is a valid quantum state (positive semidefinite and normalized), "
        "and if Z and X are Hermitian (observables)."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Assess the effect of the matrix exponential e^X on vectors and states, specifically whether multiplication by e^X changes the norm of any vector, "
        "and analyze the transformed state (e^X)*Y*(e^{-X}) to verify if it remains a valid quantum state, based on the output from Sub-task 1."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
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
        "Sub-task 3: Evaluate the correctness of each given statement (choices 1 to 4) based on the analyzed properties and transformation impacts, "
        "determining which statements are true in the quantum mechanics context, using outputs from Sub-task 1 and Sub-task 2."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', ''), results2.get('thinking', ''), results2.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3.get('thinking', ''), results3.get('answer', ''))
    return final_answer, logs
