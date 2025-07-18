async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the first reaction step: treatment of the starting material with sodium hydride followed by benzyl bromide, "
        "and determine the structure of product 1, with context from the given query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the second reaction step: treatment of product 1 with p-toluenesulfonyl hydrazide and catalytic HCl to form product 2, "
        "focusing on hydrazone formation at the ketone, with context from the given query and output of subtask 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the structure of product 2, "
        "given all the above thinking and answers from subtask 1."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
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
        "Sub-task 3: Analyze the third reaction step: treatment of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride, "
        "to determine the structural changes leading to product 3, with context from the given query and outputs of subtasks 1 and 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent answer for the structure of product 3, "
        "given all the above thinking and answers from subtasks 1 and 2."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Analyze the fourth reaction step: catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere to form product 4, "
        "focusing on reduction of functional groups and protecting groups, with context from the given query and outputs of subtasks 1, 2, and 3."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent answer for the structure of product 4, "
        "given all the above thinking and answers from subtasks 1, 2, and 3."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Integrate the structural information from products 1 to 4 to compute and confirm the final structure of product 4, "
        "and compare it with the given choices to identify the correct product, with context from the given query and outputs of subtasks 1 to 4."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions for the integration and final structure determination of product 4."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_instruction6 = (
        "Sub-task 6: Combine all previous analyses and transformations to produce the final answer: the structure of product 4 corresponding to one of the provided choices, "
        "with context from the given query and output of subtask 5."
    )
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6, log6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
