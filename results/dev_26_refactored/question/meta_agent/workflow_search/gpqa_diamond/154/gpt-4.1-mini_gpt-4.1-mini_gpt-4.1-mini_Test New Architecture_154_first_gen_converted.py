async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Verify and normalize the given state vector expressed in the P_z eigenbasis, "
        "and clearly represent the operators P_x, P_y, and P_z matrices for further calculations."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Calculate the expectation value <P_z> using the normalized state vector "
        "and the P_z operator matrix, based on the output from Sub-task 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent value for <P_z> expectation value."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Calculate the expectation value <P_z^2> by squaring the P_z operator matrix "
        "and evaluating its expectation value in the normalized state, based on Sub-task 1 output."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent value for <P_z^2> expectation value."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Compute the uncertainty ΔP_z = sqrt(<P_z^2> - <P_z>^2) using the previously calculated expectation values "
        "and identify the correct uncertainty value from the given choices, based on Sub-tasks 2 and 3 outputs."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and select the correct uncertainty ΔP_z value from the given choices."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
