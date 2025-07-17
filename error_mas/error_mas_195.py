async def forward_195(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize all given physical parameters and constants relevant to the problem, including mass m, spring constant k, amplitude A, and speed of light c, and identify the classical maximum speed expression for a harmonic oscillator."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, compute key dimensionless parameters and energy scales such as the ratio kA^2/(2m) (classical potential energy) and kA^2/(2mc^2) (potential energy normalized by rest energy), which are essential for relativistic corrections."
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

    cot_sc_instruction3 = "Sub-task 3: Combine the classical harmonic oscillator relations with relativistic energy and momentum relations to formulate an expression relating maximum speed v_max to the given parameters, incorporating relativistic corrections."
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

    cot_reflect_instruction4 = "Sub-task 4: Derive the explicit formula for the maximum speed v_max of the relativistic harmonic oscillator by applying energy conservation and relativistic kinetic energy expressions, ensuring dimensional consistency and physical plausibility."
    critic_instruction4 = "Please review the derived formula for v_max and provide its limitations."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction_5 = "Sub-task 5: Evaluate the four candidate formulas for v_max against the derived expression and physical constraints (e.g., v_max < c), identify which formula correctly represents the maximum speed, and justify the selection."
    debate_desc5 = {
        'instruction': debate_instruction_5,
        'context': ["user query", results4['thinking'], results4['answer']],
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
