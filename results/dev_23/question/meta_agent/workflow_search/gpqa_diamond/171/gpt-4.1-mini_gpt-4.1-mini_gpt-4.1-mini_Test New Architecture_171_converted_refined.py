async def forward_171(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Formulate the expression for the ratio of excited iron atom populations in star_1 and star_2 "
        "using the Boltzmann distribution under LTE, explicitly incorporating the energy difference ΔE and Boltzmann constant k_B, "
        "and clearly defining all variables to avoid ambiguity."
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

    cot_sc_instruction2 = (
        "Sub-task 2: Apply the natural logarithm to the population ratio expression derived in subtask_1 to isolate ln(2) "
        "and express it explicitly in terms of ΔE, k_B, and the temperatures T_1 and T_2, ensuring the formula is dimensionally consistent and physically meaningful."
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
        "Sub-task 3: Clarify and disambiguate the notation and physical meaning of each candidate equation provided in the query, "
        "explicitly defining how terms like (T1*T2)^2 or (T1*T2) are interpreted, and whether ΔE and k_B are included or normalized."
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

    debate_instruction4 = (
        "Sub-task 4: Perform a detailed algebraic and dimensional consistency check between the derived logarithmic expression from subtask_2 "
        "and each candidate equation clarified in subtask_3. This includes rewriting each candidate equation into the canonical form involving (1/T_2 - 1/T_1) or (T_1 - T_2)/(T_1 T_2) "
        "and verifying the presence and correct placement of ΔE and k_B."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    reflexion_instruction5 = (
        "Sub-task 5: Explicitly map the derived formula to the candidate equations by comparing them term-by-term, "
        "confirming which candidate equation exactly matches the derived expression including all physical constants and correct algebraic form."
    )
    reflexion_desc5 = {
        'instruction': reflexion_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=reflexion_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
