async def forward_195(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize all given physical information and parameters relevant to the relativistic harmonic oscillator problem, including mass m, amplitude A, spring constant k, speed of light c, and the candidate formulas for v_max."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze the relationships between the physical components, including the force law, energy considerations, and relativistic constraints, to classify the problem elements and identify key dimensionless parameters influencing v_max."
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

    cot_sc_instruction3 = "Sub-task 3: Derive the relativistic expression for the maximum speed v_max of the oscillator by applying relativistic energy conservation and Hooke's law, incorporating the rest energy and potential energy terms."
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

    debate_instruction4 = "Sub-task 4: Evaluate and compare the four candidate formulas for v_max against the derived expression and physical constraints (e.g., v_max < c), identifying which formula correctly represents the relativistic maximum speed."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"],
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
