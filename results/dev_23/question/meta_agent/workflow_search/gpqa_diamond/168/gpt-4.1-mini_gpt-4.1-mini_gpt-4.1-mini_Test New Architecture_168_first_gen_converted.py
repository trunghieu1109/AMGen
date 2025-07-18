async def forward_168(self, taskInfo):
    logs = []

    debate_instruction_0_1 = "Sub-task 1: Analyze and classify the original decay process 2A -> 2B + 2E + 2V, focusing on the particle types, their masses, and the nature of the energy spectrum of the E particles (continuous with endpoint Q), with context from the query."
    debate_desc_0_1 = {
        'instruction': debate_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_0_1, log_0_1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc_0_1,
        n_repeat=self.max_round
    )
    logs.append(log_0_1)

    debate_instruction_0_2 = "Sub-task 2: Analyze and classify the variant decay process 2A -> 2B + 2E + M, focusing on the replacement of two V particles by one massless M particle and implications for kinematics and energy distribution, using outputs from Sub-task 1."
    debate_desc_0_2 = {
        'instruction': debate_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results_0_2, log_0_2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc_0_2,
        n_repeat=self.max_round
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_3 = "Sub-task 3: Evaluate how the change from 2V to 1 massless M affects the total energy spectrum of the outgoing E particles, specifically whether the spectrum remains continuous or becomes discrete, and how the endpoint value changes, based on outputs from Sub-tasks 1 and 2."
    cot_sc_desc_1_3 = {
        'instruction': cot_sc_instruction_1_3,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results_1_3, log_1_3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc_1_3,
        n_repeat=self.max_sc
    )
    logs.append(log_1_3)

    debate_instruction_1_4 = "Sub-task 4: Compare the energy spectrum characteristics (continuity, shape, endpoint) of the original and variant decays to determine the correct description among the given choices, based on output from Sub-task 3."
    debate_desc_1_4 = {
        'instruction': debate_instruction_1_4,
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results_1_4, log_1_4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_1_4,
        n_repeat=self.max_round
    )
    logs.append(log_1_4)

    final_answer = await self.make_final_answer(results_1_4['thinking'], results_1_4['answer'])
    return final_answer, logs
