async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the first reaction step: treatment of the starting material with sodium hydride followed by benzyl bromide, and determine the structural changes, especially on the hydroxymethyl substituent."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = "Sub-task 2: Analyze the second reaction step: reaction of product 1 with p-toluenesulfonyl hydrazide in catalytic HCl, focusing on hydrazone formation at the ketone position and its structural implications."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Analyze the third reaction step: treatment of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride quench, elucidating the intermediate species formed and resulting structural changes."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_agent_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = "Sub-task 4: Analyze the fourth reaction step: catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere, determining which functional groups are reduced or removed and the final structural modifications."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=cot_agent_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Integrate the structural changes from all previous steps to deduce the structure of product 4 and select the correct product from the given choices."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
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
