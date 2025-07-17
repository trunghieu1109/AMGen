async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Calculate the mass ratio of the two stars in each binary system using the inverse ratio of their radial velocity amplitudes. "
        "Explicitly avoid arbitrary normalization of masses and ensure the mass ratio is expressed purely in terms of the observed RV amplitudes to prevent misinterpretation. "
        "Use the given RV amplitudes for system_1 and system_2 from the query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the mass ratios from Sub-task 1, derive and numerically compute the total mass (M1 + M2) of each binary system directly from the given orbital periods and radial velocity amplitudes using Kepler's third law. "
        "Use the formula for the semi-major axis a = (P / 2π) * (K1 + K2) and then compute Mtot = 4π²a³ / (G P²). "
        "Clearly define all variables and link them to observed data from the query. Perform numeric evaluation for system_1 and system_2."
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

    debate_instruction3 = (
        "Sub-task 3: Calculate the factor by which system_1 is more massive than system_2 by taking the ratio of their total masses computed in Sub-task 2. "
        "Carefully interpret the phrase 'by what factor is system_1 more massive than system_2' to confirm whether the ratio should be greater or less than 1, and avoid misinterpretation of the question wording."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Evaluate the computed mass ratio from Sub-task 3 against the provided multiple-choice options by precisely comparing numerical values. "
        "Explicitly quantify the numerical closeness of the computed ratio to each option and select the closest approximate value. "
        "Include a sanity check that confirms the ratio is consistent with physical expectations and the problem context. "
        "Prevent subjective or loose rounding and ensure the final answer accurately reflects the computed results."
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
