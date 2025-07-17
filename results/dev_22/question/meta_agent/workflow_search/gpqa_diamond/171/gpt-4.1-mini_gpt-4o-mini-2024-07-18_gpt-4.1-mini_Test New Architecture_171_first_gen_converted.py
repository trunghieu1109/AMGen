async def forward_171(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Analyze and summarize the given physical information: excitation ratio, energy difference, LTE assumption, and their implications for the population of excited states in the two stars, with context from taskInfo."
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

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze and classify the mathematical relationships involved, specifically the Boltzmann distribution and how it relates excitation ratios to temperature and energy difference, with context from taskInfo and thinking of subtask 1."
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

    debate_instruction3 = "Sub-task 3: Select and transform the physical and mathematical elements to derive an equation relating the effective temperatures T_1 and T_2 to the excitation ratio and energy difference, using logarithmic manipulation of the Boltzmann factors, based on outputs from subtasks 1 and 2."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Evaluate the candidate equations by comparing their algebraic forms to the derived temperature relationship from Sub-task 3 and identify which equation correctly represents the relationship between T_1 and T_2."
    debate_desc4 = {
        'instruction': debate_instruction4,
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
