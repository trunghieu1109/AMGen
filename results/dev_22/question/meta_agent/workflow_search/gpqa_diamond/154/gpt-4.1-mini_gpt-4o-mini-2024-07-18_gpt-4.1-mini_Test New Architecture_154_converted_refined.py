async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract the given operators Px, Py, Pz matrices and the explicit state vector. "
        "Verify rigorously that the state vector is normalized and explicitly check via matrix multiplication whether it is an eigenstate of Px with eigenvalue -hbar. "
        "If the verification fails, flag the inconsistency and halt or request correction. "
        "This step addresses the previous critical failure of accepting the eigenstate assumption without explicit verification."
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
        "Sub-task 2: Confirm the normalization of the given state vector by computing its norm and verifying it equals 1. "
        "This explicit normalization check prevents propagation of errors in expectation value calculations."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the probabilities p_{+1}, p_0, p_{-1} = |c_m|^2 from the components of the normalized state vector in the Pz eigenbasis. "
        "Perform an explicit arithmetic check that the sum of probabilities equals 1 to ensure consistency before proceeding. "
        "This step prevents the previous arithmetic slip in probability calculations."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
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

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the expectation value <Pz> = sum p_m * m * hbar, where m = +1, 0, -1 are the eigenvalues of Pz. "
        "Use the probabilities from subtask 3 and perform explicit arithmetic verification of the result."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Calculate the expectation value <Pz^2> = sum p_m * (m * hbar)^2 using the probabilities from subtask 3. "
        "Perform explicit arithmetic verification and cross-check with the sum of probabilities to avoid the previous error where <Pz^2> was underestimated."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    cot_sc_instruction6 = (
        "Sub-task 6: Compute the uncertainty Delta Pz = sqrt(<Pz^2> - <Pz>^2) using the expectation values from subtasks 4 and 5. "
        "Include an explicit check that the quantity under the square root is non-negative and interpret the result in terms of hbar. "
        "This step ensures correct final uncertainty calculation."
    )
    cot_sc_desc6 = {
        'instruction': cot_sc_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6, log6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_sc_desc6,
        n_repeat=self.max_sc
    )
    logs.append(log6)

    debate_instruction7 = (
        "Sub-task 7: Compare the computed uncertainty Delta Pz with the given multiple-choice options and select the correct answer. "
        "Justify the choice clearly based on the numerical result. This final step ensures the answer is consistent with the calculations and problem context."
    )
    debate_desc7 = {
        'instruction': debate_instruction7,
        'context': ["user query", results6['thinking'], results6['answer']],
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results7, log7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc7,
        n_repeat=self.max_round
    )
    logs.append(log7)

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'])
    return final_answer, logs
