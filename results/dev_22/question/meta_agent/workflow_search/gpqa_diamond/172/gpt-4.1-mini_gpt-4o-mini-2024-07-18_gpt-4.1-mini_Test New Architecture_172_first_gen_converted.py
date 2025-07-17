async def forward_172(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize all given physical quantities and constants needed for calculations, including electron mass, reduced Planck constant ħ, velocity v, and uncertainty in position Δx, based on the user query."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = "Sub-task 2: Calculate the minimum uncertainty in momentum Δp using the Heisenberg uncertainty principle Δx Δp ≥ ħ/2, based on extracted constants and quantities from Sub-task 1."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Relate the uncertainty in momentum Δp to the uncertainty in kinetic energy ΔE using the kinetic energy formula E = p^2/(2m) and appropriate differential approximations, based on outputs from Sub-task 1 and Sub-task 2."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = "Sub-task 4: Estimate the numerical value of the minimum uncertainty in energy ΔE using the calculated Δp and given electron velocity and mass, based on Sub-task 3."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Based on the output of Sub-task 4, compare the estimated ΔE with the provided multiple-choice options and select the closest value, justifying the choice."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
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
