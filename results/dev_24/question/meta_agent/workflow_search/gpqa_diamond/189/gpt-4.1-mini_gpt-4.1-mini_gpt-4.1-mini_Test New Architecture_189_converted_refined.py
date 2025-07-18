async def forward_189(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the given nucleophiles based on their chemical nature, charge, functional groups, "
        "and structural features. Identify the type of nucleophile (alkoxide, hydroxide, carboxylate, thiolate, alcohol) "
        "and note steric and electronic characteristics relevant to nucleophilicity. Avoid oversimplified assumptions about steric hindrance and solvation effects."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Analyze the effect of aqueous solvent on nucleophilicity for the given nucleophiles, "
        "focusing on solvation, hydrogen bonding, and steric factors. Explicitly incorporate feedback that sulfur nucleophiles (e.g., ethanethiolate) "
        "experience stronger solvation and protonation equilibria in water than previously assumed, which reduces their nucleophilicity relative to oxygen nucleophiles. "
        "Avoid overestimating nucleophilicity based on polarizability alone. Use the output from Sub-task 1 as context."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the solvent effect on nucleophilicity, "
        "given all the above thinking and answers."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Critically evaluate and compare the intrinsic nucleophilicity of each nucleophile in aqueous solution by integrating chemical nature, solvent effects, steric hindrance, and experimental trends. "
        "Challenge assumptions about sulfur nucleophile behavior in protic solvents using literature or standard reference data. Use forced counterarguments to ensure robust validation of nucleophilicity order. "
        "Use outputs from Sub-tasks 1 and 2 as context."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a robust validated nucleophilicity order for the nucleophiles in aqueous solution."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Perform a dedicated pairwise comparison between hydroxide and 4-methylcyclohexan-1-olate nucleophiles in aqueous solution. "
        "Explicitly address the previous error of mis-ranking these two species by balancing steric hindrance against solvation effects, citing experimental or theoretical evidence. "
        "Resolve this critical pairwise ranking with high confidence before proceeding. Use outputs from Sub-task 3 as context."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions for the hydroxide vs. 4-methylcyclohexan-1-olate nucleophile ranking."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_instruction5 = (
        "Sub-task 5: Prioritize and arrange all five nucleophiles from most reactive to least reactive in aqueous solution based on the validated evaluations from previous subtasks, "
        "especially incorporating the corrected hydroxide vs. 4-methylcyclohexan-1-olate ranking and solvent-specific nucleophilicity insights. "
        "Integrate all prior findings to produce a final ranked list."
    )
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking and answers of subtasks 1-4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Map the derived nucleophile reactivity order to the exact multiple-choice options provided. "
        "Perform a sanity check to ensure the final order matches one and only one of the given answer choices. "
        "If no exact match is found, flag the discrepancy and trigger a re-evaluation of the ranking. "
        "Use output from Sub-task 5 as context."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Confirm the final nucleophile reactivity order aligns with one of the provided answer choices or indicate discrepancy."
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
