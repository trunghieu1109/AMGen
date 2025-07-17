async def forward_154(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given information about the operators Px, Py, Pz and the system state vector, "
        "including matrix forms, basis, and eigenvalues, with context from the user query."
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

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, analyze the relationships between the operators and the given state, "
        "confirming the basis choice, normalization of the state vector, and physical interpretation of the operators."
    )
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Compute the expectation values <Pz> and <Pz^2> in the given state vector using the Pz matrix and the state expressed in the Pz eigenbasis, "
        "and provide detailed reasoning and calculations."
    )
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc={
            'instruction': debate_instruction3,
            'context': ["user query", results2['thinking'], results2['answer']],
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Based on the outputs from Sub-task 2 and Sub-task 3, calculate the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) "
        "using the expectation values obtained, and identify the correct uncertainty value from the given choices."
    )
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc={
            'instruction': cot_reflect_instruction4,
            'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
        },
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
