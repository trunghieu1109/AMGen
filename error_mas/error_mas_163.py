async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and compute the mass ratio of the two stars in each system using the given radial velocity amplitudes. "
        "Use the radial velocity amplitudes to find the mass ratio of the stars in system_1 and system_2 respectively, considering the inverse proportionality between velocity amplitude and mass. "
        "Provide detailed reasoning and final mass ratios for both systems."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Calculate the total mass of each binary system using the orbital period and the radial velocity amplitudes, "
        "applying Kepler's third law and orbital mechanics assumptions. Use the mass ratios from Sub-task 1 to find individual masses and sum them. "
        "Consider circular orbits and edge-on inclination due to eclipses. Provide detailed calculations and final total masses for system_1 and system_2."
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

    cot_reflect_instruction3 = (
        "Sub-task 3: Compute the ratio of the total mass of system_1 to system_2 based on the total masses derived in Sub-task 2. "
        "Review the calculations and filter valid scenarios that meet the problem conditions. Provide reasoning and final mass ratio."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction_4 = (
        "Sub-task 4: Evaluate and interpret the computed mass ratio from Sub-task 3 to select the closest approximate answer choice from the given options. "
        "Provide detailed reasoning and final selection among the choices: '~ 0.4', '~ 0.7', '~ 0.6', '~ 1.2'."
    )
    debate_desc4 = {
        'instruction': debate_instruction_4,
        'context': ["user query", results3['thinking'], results3['answer']],
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
