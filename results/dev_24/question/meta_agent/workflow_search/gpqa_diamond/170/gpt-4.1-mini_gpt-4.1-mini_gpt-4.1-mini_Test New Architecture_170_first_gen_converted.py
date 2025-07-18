async def forward_170(self, taskInfo):
    logs = []

    debate_instruction1 = (
        "Sub-task 1: Analyze and classify the substituents on the benzene ring (CH3, COOC2H5, Cl, NO2, C2H5, COOH) by their electronic nature (electron-donating or withdrawing), directing effects (ortho/para or meta), and activation/deactivation of the ring."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'input': [taskInfo],
        'context_desc': ["user query"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the classification of substituents from Sub-task 1, evaluate steric and electronic factors affecting the formation of the para-isomer in electrophilic bromination for each substituent, including potential steric hindrance and strength of activation/deactivation."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct evaluation of steric and electronic factors affecting para-isomer formation."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Integrate the classification and evaluation results from Sub-tasks 1 and 2 to predict the relative weight fraction of the para-bromo isomer formed for each substituted benzene."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a reasoned prediction of the relative para-isomer yields based on integrated analysis."
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

    cot_sc_instruction4 = (
        "Sub-task 4: Arrange the six substances in order of increasing weight fraction of the para-isomer yield based on the integrated analysis from Sub-task 3."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the ordered list of substances from lowest to highest para-isomer yield fraction."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Compare the predicted order of increasing para-isomer yield from Sub-task 4 with the given multiple-choice options and select the correct ranking."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of the predicted order and justify the selection of the correct multiple-choice option."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        'temperature': 0.0
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
