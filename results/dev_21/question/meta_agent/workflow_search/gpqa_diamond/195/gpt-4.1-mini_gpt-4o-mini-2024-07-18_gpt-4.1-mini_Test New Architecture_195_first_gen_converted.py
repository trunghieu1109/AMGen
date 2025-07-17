async def forward_195(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Extract and classify all given quantitative parameters and physical laws relevant to the relativistic harmonic oscillator, including mass m, amplitude A, spring constant k, speed of light c, and the form of Hooke's law, based on the provided query."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze the physical and mathematical relationships between the parameters, focusing on how relativistic effects modify classical harmonic oscillator dynamics and how energy considerations relate to maximum speed."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = "Sub-task 3: Derive the expression for the maximum speed v_max of the relativistic harmonic oscillator by applying relativistic energy conservation and the given force law, incorporating the relativistic kinetic energy and potential energy at maximum amplitude."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Evaluate the four candidate formulas for v_max against the derived expression, checking for physical consistency (e.g., v_max < c), dimensional correctness, and alignment with relativistic principles."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Select and prioritize the correct formula for v_max from the candidates based on the evaluation, providing justification for the choice and explaining why other options are incorrect or less appropriate."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_2.subtask_4", "answer of stage_2.subtask_4"]
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
