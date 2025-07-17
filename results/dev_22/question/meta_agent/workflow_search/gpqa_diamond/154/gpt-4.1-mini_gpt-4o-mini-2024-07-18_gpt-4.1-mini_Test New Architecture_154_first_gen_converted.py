async def forward_154(self, taskInfo):
    logs = []

    debate_instruction_1 = "Sub-task 1: Extract and verify the given operators Px, Py, Pz matrices and the explicit state vector; confirm the state vector is normalized and corresponds to the eigenvalue -ħ of Px."
    debate_desc_1 = {
        'instruction': debate_instruction_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction_2 = "Sub-task 2: Express the operator Pz in matrix form and prepare to calculate expectation values ⟨Pz⟩ and ⟨Pz²⟩ using the given state vector, based on the output from Sub-task 1."
    cot_sc_desc_2 = {
        'instruction': cot_sc_instruction_2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc_2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction_3 = "Sub-task 3: Calculate the expectation value ⟨Pz⟩ = ⟨ψ|Pz|ψ⟩ and the expectation value ⟨Pz²⟩ = ⟨ψ|Pz²|ψ⟩ using the state vector and Pz matrix, based on the output from Sub-task 2."
    debate_desc_3 = {
        'instruction': debate_instruction_3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc_3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction_4 = "Sub-task 4: Compute the uncertainty ΔPz = sqrt(⟨Pz²⟩ - ⟨Pz⟩²) based on the calculated expectation values and interpret the result in terms of ħ, based on the output from Sub-task 3."
    cot_sc_desc_4 = {
        'instruction': cot_sc_instruction_4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc_4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction_5 = "Sub-task 5: Compare the computed uncertainty ΔPz with the given multiple-choice options and select the correct answer, based on the output from Sub-task 4."
    debate_desc_5 = {
        'instruction': debate_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc_5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
