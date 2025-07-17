async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze and classify the starting material and each reagent's role and expected chemical transformation in the sequence with context from the given organic synthesis query."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = "Sub-task 2: Summarize the structural changes expected after each reaction step, focusing on functional group transformations and substituent modifications, based on the analysis from Sub-task 1."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage_0.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Evaluate the intermediate structures after each step, especially the effect of n-butyllithium treatment on the tosyl hydrazone and the implications for the final product, based on the summary from Sub-task 2."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = "Sub-task 4: Predict the structure of product 4 by applying knowledge of catalytic hydrogenation effects on the intermediate and comparing with the given product choices, based on the evaluation from Sub-task 3 and summary from Sub-task 2."
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Select the correct product structure from the given choices based on the predicted structure from Sub-task 4 and provide a rationale for the selection."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"]
    }
    results5, log5 = await self.debate(
        subtask_id="stage_1.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
