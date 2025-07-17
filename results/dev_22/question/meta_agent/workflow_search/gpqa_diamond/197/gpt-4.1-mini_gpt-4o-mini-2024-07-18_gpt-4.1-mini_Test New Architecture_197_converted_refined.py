async def forward_197(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and classify all given information, explicitly confirm that the provided stability constants "
        "(β1=9, β2=40, β3=63, β4=16) are cumulative formation constants, and verify that the blue complex corresponds to the dithiocyanato species Co(SCN)2. "
        "Embed feedback that previous errors arose from unclear assumptions about the nature of β values."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Formulate the equilibrium concentration expressions for all cobalt(II) species (free Co(II), Co(SCN)+, Co(SCN)2, Co(SCN)3-, Co(SCN)4 2-) "
        "correctly using the confirmed cumulative stability constants and the free SCN- concentration. Include a verification step to cross-check that the expressions align with the definition of cumulative constants, "
        "explicitly avoiding the previous error of treating β values as stepwise constants."
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

    cot_instruction3 = (
        "Sub-task 3: Derive the explicit mass balance equation for total cobalt concentration by summing the concentrations of free Co(II) and all complexed species expressed via cumulative constants and free SCN- concentration. "
        "Embed feedback that previous attempts failed to substitute correct expressions and solve for free Co(II), so the equation must be fully explicit and ready for numeric solution."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=cot_agent_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 1: Numerically solve the mass balance equation derived in stage_1.subtask_3 by substituting the given total cobalt concentration (10^-2 M) and SCN- concentration (0.1 M) to find the free Co(II) concentration. "
        "Then calculate the equilibrium concentrations of all cobalt species, including the dithiocyanato complex Co(SCN)2, using the verified equilibrium expressions. "
        "Avoid symbolic placeholders and ensure actual numeric evaluation to prevent previous errors of incomplete numeric solving."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 2: Compute the percentage of the dithiocyanato cobalt(II) complex Co(SCN)2 relative to the total cobalt concentration using the concentrations calculated in stage_2.subtask_1. "
        "Compare the result with the given answer choices and provide the final answer. Ensure the percentage is based on correctly computed concentrations, addressing the previous failure of guessing rather than calculating the final percentage."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
