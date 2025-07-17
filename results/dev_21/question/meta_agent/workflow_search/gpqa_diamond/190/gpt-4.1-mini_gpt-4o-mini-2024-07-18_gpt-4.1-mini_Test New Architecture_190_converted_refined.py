async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the first reaction step: treatment of the starting material with sodium hydride followed by benzyl bromide. Determine the structural changes, focusing on the conversion of the hydroxymethyl substituent to the benzyl ether. Avoid assumptions about other parts of the molecule at this stage."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = "Sub-task 2: Analyze the second reaction step: reaction of product 1 with p-toluenesulfonyl hydrazide in catalytic HCl. Focus on the formation of the tosyl hydrazone at the ketone position, detailing the structural and mechanistic implications. Ensure clear understanding of the hydrazone intermediate to support subsequent steps."
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

    cot_sc_instruction3 = "Sub-task 3: Identify and explain the reaction mechanism of the third step: treatment of the tosyl hydrazone (product 2) with n-butyllithium at low temperature. Explicitly recognize this as a Shapiro reaction, detailing the stepwise mechanism including deprotonations, elimination of nitrogen and tosyl groups, and formation of the vinyl lithium intermediate. Emphasize that this step does NOT introduce a butyl substituent, correcting previous mechanistic misunderstandings."
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

    cot_sc_instruction4 = "Sub-task 4: Predict the structural changes after aqueous ammonium chloride quench of the vinyl lithium intermediate formed in subtask_3. Determine the resulting alkene structure at the former ketone carbon, ensuring no butyl substitution is introduced. This subtask should explicitly connect the mechanistic outcome of the Shapiro reaction to the observed product structure."
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Analyze the fourth reaction step: catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere. Determine which functional groups are reduced or removed, including hydrogenation of the alkene formed in subtask_4, reduction of the isopropenyl group to isopropyl, and hydrogenolysis of the benzyl ether to regenerate the free alcohol. Avoid assumptions that the benzyl ether remains intact. Provide a detailed account of all bond changes and functional group transformations."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 1", "answer of subtask 1"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = "Sub-task 6: Integrate the structural changes and mechanistic insights from all previous subtasks (1 through 5) to deduce the final structure of product 4. Critically evaluate all transformations, ensuring no mechanistic errors from prior steps propagate. Use this integrated understanding to select the correct product from the given choices, justifying the selection based on the cumulative evidence."
    debate_desc6 = {
        'instruction': debate_instruction6,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
