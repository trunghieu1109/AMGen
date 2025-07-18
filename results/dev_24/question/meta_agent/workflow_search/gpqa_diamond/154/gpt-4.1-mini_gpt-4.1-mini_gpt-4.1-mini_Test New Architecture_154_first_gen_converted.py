async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Summarize and verify the given operators P_x, P_y, P_z matrices and the state vector psi in the P_z eigenbasis, "
        "ensuring correct interpretation of the problem setup and normalization of the state. "
        "Include explicit matrix forms and check normalization of the state vector."
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

    cot_instruction2 = (
        "Sub-task 2: Express the operator P_z and its square P_z^2 explicitly in matrix form and confirm their action on the given state vector psi. "
        "Verify the matrices and their multiplication with psi."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the expectation values <P_z> = psi^dagger P_z psi and <P_z^2> = psi^dagger P_z^2 psi using the matrices and state vector from Stage 0. "
        "Consider all calculation steps carefully and verify results for consistency."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct expectation values for <P_z> and <P_z^2> given all the above thinking and answers."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Calculate the uncertainty Delta P_z = sqrt(<P_z^2> - <P_z>^2) using the expectation values obtained in Sub-task 3. "
        "Provide detailed calculation steps and the numerical/symbolic result."
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

    debate_instruction5 = (
        "Sub-task 5: Compare the calculated uncertainty Delta P_z with the given multiple-choice options and select the correct answer. "
        "Justify the choice clearly based on the calculations."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final answer for the uncertainty Delta P_z from the given choices, based on the previous calculations."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
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
