async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the substituents on the benzene ring (CH3, COOC2H5, Cl, NO2, C2H5, COOH) by their electronic nature (electron-donating or withdrawing), directing effects (ortho/para or meta), and activation/deactivation of the ring. "
        "Include a detailed comparative analysis between similar electron-withdrawing substituents (ester vs carboxylic acid) referencing experimental or literature data to avoid misclassification errors."
    )
    debate_desc1 = {
        'instruction': cot_instruction1,
        'final_decision_instruction': 'Sub-task 1: Provide a detailed classification and comparative analysis of the substituents as described.',
        'input': [taskInfo],
        'context_desc': ['user query'],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id='subtask_1',
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the classification from Subtask 1, estimate the approximate mole-fraction selectivity (% para product by mole) for the electrophilic bromination of each substituted benzene. "
        "Explicitly track units (moles vs grams) and consider steric and electronic factors affecting para substitution. "
        "Address previous errors conflating mole fraction with weight fraction and incorrect assumptions about relative para selectivity of methyl vs ethyl substituents."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent mole-fraction para-selectivity estimates for each substituent, given all prior analysis."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1'],
        'temperature': 0.5
    }
    results2, log2 = await self.sc_cot(
        subtask_id='subtask_2',
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Convert the mole-fraction para-selectivities obtained in Subtask 2 into weight fractions by multiplying the mole fractions by the molecular masses of the corresponding para-bromo derivatives. "
        "Ensure correct ranking of substituents based on weight fraction of para-isomer yield, explicitly addressing molecular weight effects."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the weight-fraction para-isomer yields for each substituent, consistent with mole-fraction data and molecular weights."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ['user query', 'thinking of subtask 2', 'answer of subtask 2'],
        'temperature': 0.5
    }
    results3, log3 = await self.sc_cot(
        subtask_id='subtask_3',
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Integrate the classification results (Subtask 1) and the calculated weight-fraction para-isomer yields (Subtask 3) to arrange the six substances in order of increasing weight fraction of the para-isomer yield. "
        "Cross-validate the order against known chemical principles and experimental trends, especially verifying the relative positions of ester vs carboxylic acid and methyl vs ethyl substituents to avoid misrankings."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the ordered list of substances by increasing weight fraction of para-isomer yield, with justification."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 3', 'answer of subtask 3'],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id='subtask_4',
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Critically evaluate and reflexively review the proposed ranking from Subtask 4. "
        "Challenge any inconsistencies or contradictions with established aromatic substitution principles and experimental data. "
        "Detect and correct errors before final answer selection, improving reliability and avoiding propagation of subtle mistakes."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of the proposed ranking and suggest corrections if needed."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results4['thinking'], results4['answer']],
        'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 4', 'answer of subtask 4'],
        'temperature': 0.0
    }
    results5, log5 = await self.reflexion(
        subtask_id='subtask_5',
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_instruction6 = (
        "Sub-task 6: Compare the final validated order of substances by increasing weight fraction of para-isomer yield with the given multiple-choice options and select the correct ranking. "
        "Ensure the choice is consistent with all prior analysis and corrections."
    )
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 5', 'answer of subtask 5']
    }
    results6, log6 = await self.cot(
        subtask_id='subtask_6',
        cot_agent_desc=cot_agent_desc6
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
