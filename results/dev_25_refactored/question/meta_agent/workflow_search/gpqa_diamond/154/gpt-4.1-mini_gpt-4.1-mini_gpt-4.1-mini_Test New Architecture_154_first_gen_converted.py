async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Verify and normalize the given state vector; confirm the matrix forms of Px, Py, and Pz operators and their basis, "
        "with context from the user query."
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
        "Sub-task 2: Calculate the expectation value <Pz> using the normalized state vector and the Pz operator matrix, "
        "based on the output from Sub-task 1 and the user query."
    )
    final_decision_instruction2a = (
        "Sub-task 2: Synthesize and choose the most consistent value for <Pz> expectation value calculation."
    )
    cot_sc_desc2a = {
        "instruction": cot_sc_instruction2a,
        "final_decision_instruction": final_decision_instruction2a,
        "input": [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2a, log2a = await self.sc_cot(
        subtask_id="subtask_2a",
        cot_agent_desc=cot_sc_desc2a,
        n_repeat=self.max_sc
    )
    logs.append(log2a)

    cot_sc_instruction2b = (
        "Sub-task 3: Calculate the expectation value <Pz^2> using the normalized state vector and the square of the Pz operator matrix, "
        "based on the output from Sub-task 1 and the user query."
    )
    final_decision_instruction2b = (
        "Sub-task 3: Synthesize and choose the most consistent value for <Pz^2> expectation value calculation."
    )
    cot_sc_desc2b = {
        "instruction": cot_sc_instruction2b,
        "final_decision_instruction": final_decision_instruction2b,
        "input": [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2b, log2b = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc2b,
        n_repeat=self.max_sc
    )
    logs.append(log2b)

    debate_instruction4 = (
        "Sub-task 4: Compute the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) using the expectation values from Sub-tasks 2 and 3, "
        "and compare the result with the given choices to select the correct uncertainty value."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the correct uncertainty ΔPz value from the given choices based on calculations."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results2a.get('thinking', ''), results2a.get('answer', ''), results2b.get('thinking', ''), results2b.get('answer', '')],
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4.get('thinking', ''), results4.get('answer', ''))
    return final_answer, logs
