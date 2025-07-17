async def forward_163(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Compute the mass ratio (m1/m2) of the two stars in each binary system "
        "using the inverse ratio of their radial velocity amplitudes. Provide accurate mass ratio inputs "
        "for subsequent total mass calculations, with context from taskInfo."
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

    debate_instruction_2a = (
        "Sub-task 2a: Derive symbolically the formula for the total mass (M1 + M2) of a circular, edge-on binary system "
        "in terms of the orbital period P and the sum of radial velocity semi-amplitudes (K1 + K2), explicitly showing that "
        "(M1 + M2) = (P / 2πG) * (K1 + K2)^3. Incorporate Kepler's third law and orbital velocity relations, "
        "addressing previous errors, with context from taskInfo and output of subtask_1."
    )
    debate_desc_2a = {
        'instruction': debate_instruction_2a,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1"]
    }
    results2a, log2a = await self.debate(
        subtask_id="subtask_2a",
        debate_desc=debate_desc_2a,
        n_repeat=self.max_round
    )
    logs.append(log2a)

    cot_sc_instruction_2b = (
        "Sub-task 2b: Apply the derived formula from Subtask 2a to compute the numerical total masses of system_1 and system_2 "
        "using their given orbital periods and radial velocity amplitudes. Ensure correct unit consistency and avoid arbitrary units, "
        "with context from taskInfo and outputs of subtask_1 and subtask_2a."
    )
    cot_sc_desc_2b = {
        'instruction': cot_sc_instruction_2b,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2a['thinking'], results2a['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2a", "answer of subtask_2a"]
    }
    results2b, log2b = await self.sc_cot(
        subtask_id="subtask_2b",
        cot_agent_desc=cot_sc_desc_2b,
        n_repeat=self.max_sc
    )
    logs.append(log2b)

    cot_sc_instruction_3 = (
        "Sub-task 3: Compare the total masses of system_1 and system_2 computed in Stage 1 to determine the factor by which system_1 is more massive than system_2. "
        "Base comparison on physically derived total masses, not raw velocity sums or mass ratios, with context from taskInfo and outputs of subtask_2b."
    )
    cot_sc_desc_3 = {
        'instruction': cot_sc_instruction_3,
        'input': [taskInfo, results2b['thinking'], results2b['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_2b", "answer of subtask_2b"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc_3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction_4 = (
        "Sub-task 4: Select the closest approximate answer choice for the mass factor of system_1 relative to system_2 based on the comparison in Subtask 3. "
        "Include a Reflexion step to cross-check reasoning, verify correctness of formulas and calculations, and ensure no errors propagate, "
        "with context from taskInfo and outputs of subtask_3."
    )
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask_3", "answer of subtask_3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=debate_desc_4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer, log_final = await self.debate(
        subtask_id="final_answer_selection",
        debate_desc={
            'instruction': "Finalize the answer selection based on reflexion output, providing the final answer choice concisely.",
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'output': ['thinking', 'answer'],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask_4", "answer of subtask_4"]
        },
        n_repeat=1
    )
    logs.append(log_final)

    answer = await self.make_final_answer(final_answer['thinking'], final_answer['answer'])
    return answer, logs
