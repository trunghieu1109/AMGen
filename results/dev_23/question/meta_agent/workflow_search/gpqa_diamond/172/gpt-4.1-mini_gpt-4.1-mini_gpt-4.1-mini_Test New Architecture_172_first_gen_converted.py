async def forward_172(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and convert all given numerical data and constants needed for calculations, including electron mass, Planck's constant, Δx, and electron speed v, with context from taskInfo"
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

    cot_sc_instruction2 = "Sub-task 2: Calculate the minimum uncertainty in momentum Δp using the Heisenberg uncertainty principle Δx Δp ≥ ħ/2, based on outputs from Sub-task 1"
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

    cot_instruction3 = "Sub-task 3: Relate the uncertainty in momentum Δp to the uncertainty in kinetic energy ΔE using the kinetic energy formula and appropriate approximations, based on outputs from Sub-task 1 and Sub-task 2"
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Evaluate whether relativistic corrections are necessary given the electron speed and decide on the appropriate kinetic energy formula to use, based on outputs from Sub-task 1"
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Estimate the minimum uncertainty in energy ΔE numerically and compare it with the given multiple-choice options to select the closest value, based on outputs from Sub-task 3 and Sub-task 4"
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
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
