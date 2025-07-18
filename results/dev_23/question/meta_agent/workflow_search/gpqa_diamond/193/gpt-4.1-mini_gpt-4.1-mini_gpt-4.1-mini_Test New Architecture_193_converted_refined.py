async def forward_193(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Enumerate all possible spin configurations (S1, S2, S3), each ±1, "
        "and calculate the corresponding energy E = -J (S1 S2 + S1 S3 + S2 S3) for each configuration. "
        "Provide a detailed list of configurations and their energies."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, group the spin configurations by their energy values, "
        "determine the degeneracy (number of configurations) for each distinct energy level, "
        "and explicitly list these degeneracies and energies."
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
        "Sub-task 3: Derive the partition function Z as the sum over energy levels: "
        "Z = sum(degeneracy * exp(-beta * E)), using the grouped energies and degeneracies from Sub-task 2. "
        "Show the derivation explicitly and clearly."
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

    cot_reflect_instruction4 = (
        "Sub-task 4: Explicitly parse and compare the derived partition function formula term-by-term "
        "with each given multiple-choice option. Extract coefficients and exponents from each choice, "
        "match them precisely with the derived degeneracies and energy exponents, and identify the exact matching choice. "
        "Avoid majority voting or assumptions and use structured, stepwise verification."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
