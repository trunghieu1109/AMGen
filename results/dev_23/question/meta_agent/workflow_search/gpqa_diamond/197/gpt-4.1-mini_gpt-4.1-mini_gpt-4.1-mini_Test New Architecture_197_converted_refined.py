async def forward_197(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given numeric data (concentrations, stability constants) and explicitly verify the identity of the 'blue dithiocyanato cobalt(II) complex' by analyzing the stability constants and chemical properties to avoid misidentification. "
        "This subtask addresses the previous failure of confusing the dithiocyanato complex with the tetrathiocyanato species and ensures the correct target species is identified before proceeding."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Formulate the mass-balance and equilibrium expressions for all cobalt(II) thiocyanato species using the cumulative stability constants and given concentrations. "
        "Numerically solve these equations to determine the free Co(II) concentration and verify or adjust the assumption that free SCN- concentration remains approximately 0.1 M. "
        "This subtask directly addresses the previous failure to perform actual numerical calculations and ensures the equilibrium concentrations are quantitatively determined."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 1: Calculate the concentration of the blue dithiocyanato complex (Co(SCN)2) using the free Co(II) concentration and stability constants obtained from the previous subtask. "
        "Compute its percentage relative to the total cobalt concentration. This subtask ensures the numerical calculation of the target species concentration and percentage, avoiding the previous omission of plugging in numbers."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 2: Compare the calculated percentage of the blue dithiocyanato complex with the provided answer choices and select the closest matching option. "
        "This final step ensures the solution is connected to the multiple-choice format and confirms the correctness of the numerical results."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
