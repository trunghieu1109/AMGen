async def forward_166(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Construct the normalized Schrödinger cat state |psi> for phi = -pi/4 and alpha = 0.5, "
        "compute its density matrix rho = |psi><psi|, and explicitly calculate normalization constant N. "
        "Provide explicit expressions and numerical values for |psi> and rho."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2a = (
        "Sub-task 2a: Compute the first moments (displacement vector) and second moments (covariance matrix) "
        "of the non-Gaussian state rho obtained in Subtask 1, with explicit formulas and numerical evaluation."
    )
    final_decision_instruction2a = (
        "Sub-task 2a: Synthesize and choose the most consistent numeric results for the moments of rho."
    )
    cot_sc_desc2a = {
        "instruction": cot_sc_instruction2a,
        "final_decision_instruction": final_decision_instruction2a,
        "input": [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2a, log2a = await self.sc_cot(
        subtask_id="subtask_2a",
        cot_agent_desc=cot_sc_desc2a,
        n_repeat=self.max_sc
    )
    logs.append(log2a)

    cot_sc_instruction2b = (
        "Sub-task 2b: Construct the reference Gaussian state tau by defining its covariance matrix and displacement vector "
        "to match the moments computed in Subtask 2a. Provide explicit matrix forms and numerical values, clarifying assumptions about purity or mixedness."
    )
    final_decision_instruction2b = (
        "Sub-task 2b: Synthesize and finalize the explicit form and numeric values of tau's covariance matrix and displacement vector."
    )
    cot_sc_desc2b = {
        "instruction": cot_sc_instruction2b,
        "final_decision_instruction": final_decision_instruction2b,
        "input": [taskInfo, results2a.get('thinking', ''), results2a.get('answer', '')],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 2a", "answer of subtask 2a"]
    }
    results2b, log2b = await self.sc_cot(
        subtask_id="subtask_2b",
        cot_agent_desc=cot_sc_desc2b,
        n_repeat=self.max_sc
    )
    logs.append(log2b)

    debate_instruction3 = (
        "Sub-task 3: Compute the von Neumann entropy S(tau) of the reference Gaussian state tau by: "
        "(1) calculating symplectic eigenvalues of tau's covariance matrix, "
        "(2) applying the Gaussian entropy formula S(tau) = sum_k h(nu_k), "
        "and (3) performing numerical evaluation. Confirm S(rho) = 0 since rho is pure. "
        "Provide fully grounded numeric results."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and finalize the numeric value of S(tau) and confirm S(rho) = 0."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2b.get('thinking', ''), results2b.get('answer', ''), results1.get('thinking', ''), results1.get('answer', '')],
        "context_desc": ["user query", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Calculate the relative entropy measure of non-Gaussianity del_b = trace(rho ln rho) - trace(tau ln tau) = 0 - S(tau) "
        "using the entropy computed in Subtask 3. Interpret the result carefully and verify numerical consistency."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and finalize the numeric value of del_b with interpretation."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3.get('thinking', ''), results3.get('answer', '')],
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Compare the calculated non-Gaussianity value del_b with the provided answer choices (2.48, 0, 1.38, 0.25) "
        "and select the correct answer. Justify the selection based on rigorous calculations."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Select and justify the correct answer choice based on computed del_b."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4.get('thinking', ''), results4.get('answer', '')],
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5.get('thinking', ''), results5.get('answer', ''))
    return final_answer, logs
