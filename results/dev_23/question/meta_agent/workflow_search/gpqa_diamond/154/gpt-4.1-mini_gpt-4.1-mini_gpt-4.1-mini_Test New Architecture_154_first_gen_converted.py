async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and verify the given operators Px, Py, and Pz matrices and the explicit state vector in the Pz eigenbasis; "
        "confirm normalization and interpret the physical context with context from taskInfo."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Compute the expectation value <Pz> and <Pz^2> for the given state vector using the Pz operator matrix, "
        "based on the output from Sub-task 1, with context from taskInfo and results1."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1"],
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    reflexion_instruction3 = (
        "Sub-task 3: Calculate the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) using the expectation values obtained, "
        "and identify the correct uncertainty value from the given choices, based on outputs from Sub-tasks 1 and 2."
    )
    critic_instruction3 = (
        "Please review the calculation of ΔPz and the selection of the correct choice, providing any limitations or corrections needed."
    )
    cot_reflect_desc3 = {
        'instruction': reflexion_instruction3,
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', ''), results2.get('thinking', ''), results2.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3.get('thinking', ''), results3.get('answer', ''))
    return final_answer, logs
