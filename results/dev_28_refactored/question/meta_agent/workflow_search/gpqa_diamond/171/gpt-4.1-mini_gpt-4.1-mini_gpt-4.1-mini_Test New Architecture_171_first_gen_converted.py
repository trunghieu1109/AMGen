async def forward_171(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Derive the quantitative relationship between the excitation ratio (2) and the temperatures T_1 and T_2 "
        "using the Boltzmann distribution and given energy difference ΔE, with context from the user query."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent derivation for the excitation ratio-temperature relationship.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, construct the intermediate mathematical expression relating ln(2) to ΔE, Boltzmann constant k, and temperatures T_1 and T_2."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent intermediate expression relating ln(2) and temperatures.",
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Apply algebraic transformations to the intermediate expression from Sub-task 2 to isolate and simplify the relationship between ln(2) and T_1, T_2, "
        "ensuring the form matches one of the candidate equations."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': "Sub-task 3: Synthesize and choose the most consistent simplified temperature relationship.",
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the candidate equations against the derived and transformed expression from Sub-task 3 to identify which equation correctly represents the relationship between T_1 and T_2."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': "Sub-task 4: Decide which candidate equation correctly represents the temperature relationship.",
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
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
