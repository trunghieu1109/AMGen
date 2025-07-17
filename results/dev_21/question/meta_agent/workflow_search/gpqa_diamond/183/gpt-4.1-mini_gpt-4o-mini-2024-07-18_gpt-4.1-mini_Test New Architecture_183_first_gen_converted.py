async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each reagent and reaction step in the given options according to their chemical function (e.g., Friedel-Crafts alkylation, nitration, reduction, diazotization, sulfonation, hydrolysis, nucleophilic substitution) "
        "and their typical regioselectivity and directing effects on benzene, with context from taskInfo."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, summarize the expected positional effects of substituents introduced by each reaction step, "
        "focusing on how the tert-butyl, nitro, and ethoxy groups can be installed in the correct positions (1, 2, and 3) on the benzene ring, considering directing effects and blocking strategies."
    )
    results2, log2 = await self.debate(
        subtask_id="stage_0.subtask_2",
        debate_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        },
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate each provided reaction sequence option against the target molecule’s substitution pattern and yield considerations, "
        "identifying which sequence logically and efficiently leads to 2-(tert-butyl)-1-ethoxy-3-nitrobenzene with high yield, based on outputs from stage_0.subtask_1 and stage_0.subtask_2."
    )
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc={
            'instruction': debate_instruction3,
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        },
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Identify and explain the role of sulfonation/desulfonation steps and the order of nitration and alkylation in controlling regioselectivity and preventing side reactions in the synthesis sequence, "
        "based on outputs from stage_0.subtask_2 and stage_1.subtask_3."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Select the optimal reaction sequence from the given options based on the analysis and evaluation, "
        "providing a clear rationale for why this sequence leads to the high-yield synthesis of the target compound starting from benzene, "
        "using outputs from stage_1.subtask_3 and stage_1.subtask_4."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"]
    }
    results5, log5 = await self.debate(
        subtask_id="stage_1.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
