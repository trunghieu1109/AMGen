async def forward_190(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Apply the first transformation: treat the starting material with sodium hydride and benzyl bromide to determine the structure of product 1, focusing on the site of alkylation and resulting functional groups."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for product 1 structure after alkylation."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Apply the second transformation: treat product 1 with p-toluenesulfonyl hydrazide and catalytic HCl to form product 2, identifying the hydrazone formation and its structural implications."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for product 2 structure after hydrazone formation."
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
        "Sub-task 3: Apply the third transformation: treat product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride quench to form product 3, analyzing the Shapiro reaction mechanism and resulting structural changes."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent answer for product 3 structure after Shapiro reaction."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Apply the fourth transformation: stir product 3 with Pd/C under hydrogen atmosphere to form product 4, determining the effect of catalytic hydrogenation on the molecule, including reduction of double bonds and removal of benzyl protecting groups."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent answer for product 4 structure after catalytic hydrogenation."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Combine and integrate the structural information from all transformation steps to deduce the complete structure of product 4, considering all functional group changes, stereochemistry, and substituent modifications."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Deduce the final structure of product 4 based on all previous transformations and analyses."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Evaluate the given multiple-choice options against the deduced structure of product 4 to select the correct product structure that matches all transformations and reaction outcomes."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Select the correct multiple-choice option that corresponds to the deduced product 4 structure."
    )
    debate_desc6 = {
        'instruction': debate_instruction6,
        'final_decision_instruction': final_decision_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'context_desc': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'temperature': 0.5
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
