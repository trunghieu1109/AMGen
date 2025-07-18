async def forward_171(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the given physical information and assumptions, including the excitation ratio, energy difference, and LTE condition, "
        "and express the excitation ratio in terms of the Boltzmann distribution formula."
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

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, derive the mathematical relationship between the excitation ratio (2), energy difference (ΔE), "
        "Boltzmann constant (k), and the effective temperatures T_1 and T_2 using the Boltzmann distribution under LTE."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the derived mathematical relationship between excitation ratio and temperatures."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Based on the output from Sub-task 2, manipulate the derived relationship algebraically to isolate and express ln(2) as a function of T_1 and T_2, "
        "preparing for comparison with the candidate equations."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent algebraic expression for ln(2) in terms of T_1 and T_2."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate each candidate equation against the derived expression for ln(2) to identify which correctly represents the relationship between T_1 and T_2 "
        "consistent with the Boltzmann excitation ratio."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Decide which candidate equation correctly represents the relationship between T_1 and T_2 based on the derived expression."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
