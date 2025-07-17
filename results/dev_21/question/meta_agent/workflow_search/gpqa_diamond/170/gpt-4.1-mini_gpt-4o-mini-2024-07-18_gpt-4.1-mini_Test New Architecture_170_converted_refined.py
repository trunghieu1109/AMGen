async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each substituent on the benzene ring according to authoritative chemical principles, "
        "explicitly determining their electronic effects (electron-donating or electron-withdrawing) and directing influence (ortho/para or meta) on electrophilic substitution. "
        "Explicitly cite standard chemical literature or tables to justify each classification, with special emphasis on correctly identifying the ester substituent (–COOC2H5) as a meta director due to resonance electron withdrawal. "
        "Avoid assumptions and ensure no misclassification occurs."
    )
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

    cot_reflect_instruction2 = (
        "Sub-task 2: Perform a verification and critical review of the substituent classifications from Subtask 1, "
        "focusing on identifying and flagging any uncommon, counter-intuitive, or potentially incorrect assignments (e.g., ester directing effects). "
        "Iteratively reassess and confirm the correctness of the classifications before proceeding. "
        "Prevent propagation of fundamental errors by ensuring all substituent effects are accurately and confidently assigned."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Analyze steric effects of the substituents on the benzene ring and their influence on the regioselectivity of electrophilic substitution, "
        "particularly on the relative yields of ortho and para isomers. Integrate these steric considerations with the verified electronic directing effects from Subtasks 1 and 2 "
        "to produce a comprehensive prediction of regioselectivity outcomes."
    )
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

    debate_instruction4 = (
        "Sub-task 4: Synthesize the verified substituent classifications and steric effect analyses to predict the relative weight fractions of the para-isomer yields for each of the six substances. "
        "Based on this integrated reasoning, arrange the substances in order of increasing para-isomer yield. "
        "Explicitly avoid oversimplifications and incorporate both electronic and steric factors, ensuring the final ranking is consistent with established organic chemistry principles and experimental trends."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
