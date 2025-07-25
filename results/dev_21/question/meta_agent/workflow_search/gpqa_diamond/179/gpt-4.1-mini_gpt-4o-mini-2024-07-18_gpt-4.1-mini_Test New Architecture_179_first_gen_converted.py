async def forward_179(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Compute fundamental physical constants and parameters: determine the charge value (2e), "
        "the elementary charge e, and the fixed distances (2 m) relevant for energy calculations, "
        "with context from the user query."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Calculate the electrostatic potential energy contributions between the central charge and each of the 12 charges on the sphere, "
        "using Coulomb's law and the constants computed in Sub-task 1, with context from previous outputs and the user query."
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
        "Sub-task 3: Analyze the pairwise electrostatic potential energy among the 12 charges constrained on the sphere, "
        "considering their minimal energy configuration (likely symmetric distribution), "
        "debate the possible configurations and their energy contributions with context from Sub-task 1 and the user query."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Combine the energy contributions from the central charge interactions (Sub-task 2) "
        "and the 12 charges' mutual interactions (Sub-task 3) to compute the total electrostatic potential energy of the system, "
        "debate the correctness and consistency of the combined result with context from previous subtasks and the user query."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate and identify the minimum total energy configuration, verify the result's correctness, "
        "and match it to the closest given choice rounded to three decimals, debating the final selection with context from Sub-task 4 and the user query."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
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
