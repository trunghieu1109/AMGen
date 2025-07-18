async def forward_175(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Normalize the initial state vector explicitly by computing its norm and dividing each component by the norm. "
        "This ensures correct probability calculations downstream. Avoid assumptions about normalization status. "
        "Given the initial state vector and context from taskInfo."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Provide the normalized initial state vector as a numeric list or array.",
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Compute the eigenvalues and eigenvectors of operator P by solving the characteristic polynomial det(P - lambda*I) = 0. "
        "Identify and verify the eigenspace corresponding to eigenvalue 0, including dimension and explicit normalized basis vectors. "
        "Perform numeric verification to avoid incorrect assumptions. Use the matrix P as given in taskInfo."
    )
    critic_instruction2 = (
        "Please review and provide limitations or errors in the eigenvalue and eigenvector computations for operator P, especially for eigenvalue 0 eigenspace."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'critic_instruction': critic_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Project the normalized initial state vector (from subtask_1) onto the eigenspace of P corresponding to eigenvalue 0 using the explicit normalized basis vectors found in subtask_2. "
        "Compute the projection vector numerically by inner products and linear combinations. Verify correctness before normalization. "
        "Do not rely on generic projection formulas or assumptions."
    )
    critic_instruction3 = (
        "Please review the projection vector calculation for correctness and numerical stability."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Normalize the projected post-measurement state vector obtained in subtask_3 by computing its norm and dividing each component accordingly. "
        "Ensure numeric verification to avoid zero vector or invalid normalization. "
        "This normalized vector represents the system state immediately after measuring P=0."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': "Sub-task 4: Provide the normalized post-measurement state vector as a numeric list or array.",
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Calculate the probability of measuring eigenvalue -1 for operator Q on the normalized post-measurement state vector from subtask_4. "
        "Project the post-measurement state onto the eigenvector of Q corresponding to eigenvalue -1 (known from Q's diagonal form), compute the inner product, and square its magnitude. "
        "Perform explicit numeric calculations and verify the final probability rigorously."
    )
    critic_instruction5 = (
        "Please review the probability calculation for correctness and numerical accuracy."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
