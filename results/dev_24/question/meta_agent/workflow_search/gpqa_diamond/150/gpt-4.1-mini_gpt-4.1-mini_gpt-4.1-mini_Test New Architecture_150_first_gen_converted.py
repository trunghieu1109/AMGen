async def forward_150(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Compute the eigenvalues and eigenvectors of the observable matrix P to identify the eigenspace corresponding to eigenvalue 0. "
        "Provide detailed reasoning and results for the eigen decomposition of P given in the query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc={
            'instruction': cot_instruction1,
            'final_decision_instruction': "Sub-task 1: Finalize eigenvalues and eigenvectors of P, focusing on eigenvalue 0 eigenspace.",
            'input': [taskInfo],
            'context_desc': ["user query"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Normalize the given system state vector to unit norm for proper probability calculation. "
        "Use self-consistency to ensure the normalization is correct and consistent."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent normalized state vector. "
        "Confirm the normalization factor and normalized vector components."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo],
        'temperature': 0.3,
        'context_desc': ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Project the normalized state vector onto the eigenspace associated with eigenvalue 0 using the eigenvectors found in Sub-task 1. "
        "Use debate to discuss the projection method and results."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc={
            'instruction': cot_instruction3,
            'final_decision_instruction': "Sub-task 3: Finalize the projection vector onto eigenvalue 0 eigenspace.",
            'input': [taskInfo, results1['thinking'], results2['answer']],
            'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 2"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the squared magnitude (norm squared) of the projection vector obtained in Sub-task 3 to determine the probability of measuring eigenvalue 0. "
        "Use self-consistency to ensure the probability calculation is accurate and consistent."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and confirm the calculated probability value for measuring eigenvalue 0."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.3,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Compare the calculated probability from Sub-task 4 with the given multiple-choice options and select the correct answer. "
        "Use debate to discuss and finalize the best matching choice."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Finalize the correct multiple-choice answer corresponding to the calculated probability."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
