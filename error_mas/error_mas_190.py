async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the first reaction step: treatment of the starting material with sodium hydride and benzyl bromide to determine the structural change leading to product 1, with context from the given organic synthesis query."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Analyze the second reaction step: treatment of product 1 with p-toluenesulfonyl hydrazide and catalytic HCl to form product 2, focusing on hydrazone formation, based on output from Sub-task 1 and the original query."
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = "Sub-task 3: Analyze the third reaction step: treatment of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride to form product 3, interpreting the Shapiro reaction or related transformations, based on outputs from Sub-task 2 and the original query."
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc={
            'instruction': cot_sc_instruction3,
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = "Sub-task 4: Analyze the fourth reaction step: catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere to form product 4, focusing on reduction of double bonds and removal of benzyl protecting groups, based on outputs from Sub-task 3 and the original query."
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc={
            'instruction': cot_sc_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = "Sub-task 5: Integrate the structural changes from all four reaction steps to deduce the overall transformation from the starting material to product 4, constructing the final molecular structure, based on outputs from Sub-tasks 1 to 4 and the original query."
    critic_instruction5 = "Please review the integrated structural deduction and provide its limitations."
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc={
            'instruction': cot_reflect_instruction5,
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
        },
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = "Sub-task 6: Evaluate the four given multiple-choice options against the deduced structure of product 4 to select the correct final product, based on output from Sub-task 5 and the original query."
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc={
            'instruction': debate_instruction6,
            'context': ["user query", results5['thinking'], results5['answer']],
            'input': [taskInfo, results5['thinking'], results5['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
