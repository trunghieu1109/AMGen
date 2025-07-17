async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Compute the mass ratio of the two stars in each system using the ratio of their radial velocity amplitudes. "
        "Use the given radial velocity amplitudes for system_1 (10 km/s and 5 km/s) and system_2 (15 km/s and 10 km/s) to find the mass ratios m1/m2 for each system. "
        "Explain the reasoning step-by-step with context from the query."
    )
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

    cot_sc_instruction2 = (
        "Sub-task 2: Calculate the total mass of each binary system using the orbital period and radial velocity amplitudes, "
        "applying Kepler's third law and orbital velocity relations. Use the mass ratios from Sub-task 1. "
        "Provide detailed calculations and reasoning with context from the query and Sub-task 1 outputs."
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
        "Sub-task 3: Analyze and compare the total masses of system_1 and system_2 derived from Stage 0 to understand their relative mass scale. "
        "Explain the comparison clearly and provide reasoning with context from previous subtasks."
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

    debate_instruction_4 = (
        "Sub-task 4: Derive the factor by which system_1 is more massive than system_2 by taking the ratio of their total masses "
        "and select the closest approximate answer choice from the given options (~0.4, ~0.7, ~0.6, ~1.2). "
        "Provide a reasoned debate considering uncertainties and final selection."
    )
    debate_desc4 = {
        'instruction': debate_instruction_4,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
