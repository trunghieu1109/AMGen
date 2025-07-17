async def forward_193(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Enumerate all possible spin configurations (S1, S2, S3) and compute the corresponding energy E "
        "for each configuration using E = -J(S1S2 + S1S3 + S2S3)."
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

    cot_sc_instruction2 = (
        "Sub-task 2: Calculate the Boltzmann factor e^{-βE} for each configuration and group configurations by their energy values to determine degeneracies, "
        "based on the output from Sub-task 1."
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

    debate_instruction3 = (
        "Sub-task 3: Sum the Boltzmann factors over all configurations to obtain the partition function Z expressed as a sum of exponentials with coefficients representing degeneracies, "
        "and compare the derived expression for Z with the given multiple-choice options to identify the correct partition function, "
        "based on outputs from Sub-task 2."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results2['thinking'], results2['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
