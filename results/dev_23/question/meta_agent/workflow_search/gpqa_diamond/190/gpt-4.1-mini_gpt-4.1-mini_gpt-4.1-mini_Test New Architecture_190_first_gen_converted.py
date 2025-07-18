async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the starting material and the first reaction step (treatment with NaH and benzyl bromide) "
        "to determine the structure of product 1, focusing on the protection of the hydroxymethyl group as a benzyl ether."
    )
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

    cot_instruction2 = (
        "Sub-task 2: Analyze the second reaction step (treatment of product 1 with p-toluenesulfonyl hydrazide and catalytic HCl) "
        "to determine the structure of product 2, focusing on hydrazone formation at the ketone position."
    )
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

    cot_instruction3 = (
        "Sub-task 3: Analyze the third reaction step (treatment of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride) "
        "to determine the structure of product 3, focusing on the Shapiro reaction or related base-induced transformation of the hydrazone."
    )
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

    cot_instruction4 = (
        "Sub-task 4: Analyze the fourth reaction step (stirring product 3 with Pd/C under hydrogen atmosphere) "
        "to determine the structure of product 4, focusing on catalytic hydrogenation effects such as saturation of double bonds and possible removal of benzyl protecting groups."
    )
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

    cot_sc_instruction5 = (
        "Sub-task 5: Integrate the structural information and transformations from products 1 through 4 to select the correct final product structure "
        "from the given choices, applying multi-criteria evaluation including functional group presence, protecting group status, and expected reaction outcomes."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of subtask 1", "answer of subtask 1",
            "thinking of subtask 2", "answer of subtask 2",
            "thinking of subtask 3", "answer of subtask 3",
            "thinking of subtask 4", "answer of subtask 4"
        ]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
