async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Identify and explicitly name each substituent attached to the benzene ring in the six given compounds, "
        "ensuring correct structural interpretation (e.g., distinguishing ester groups from ethers). "
        "Cross-check the molecular formula with standard chemical nomenclature and provide the correct common or IUPAC name for each substituent. "
        "Use debate agent collaboration to ensure accuracy."
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

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the identified substituents from Sub-task 1, classify each substituent based on its electronic effects "
        "(activating or deactivating) and directing effects (ortho/para or meta) in electrophilic aromatic substitution reactions. "
        "Cross-verify classifications against authoritative organic chemistry references or databases to ensure chemical accuracy, especially confirming that esters are deactivating meta directors. "
        "Use Self-Consistency Chain-of-Thought agent collaboration to validate classifications."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Analyze the influence of each correctly classified substituent on the regioselectivity of bromination, "
        "focusing on the expected ratio of para- to ortho-isomers and the overall yield of the para-isomer. "
        "Consider steric hindrance and electronic factors affecting stability and formation rates of para-substituted products. "
        "Leverage verified substituent classifications from Sub-task 2. Use debate agent collaboration to ensure thorough analysis."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Evaluate and compare the relative weight fractions of the para-isomer formed from each substituted benzene based on the regioselectivity analysis and steric considerations from Sub-task 3. "
        "Synthesize data to produce a ranked order of compounds by increasing para-isomer yield. "
        "Ensure consistency with verified substituent effects and avoid assumptions that led to prior mistakes. "
        "Use Self-Consistency Chain-of-Thought agent collaboration."
    )
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

    cot_reflect_instruction5 = (
        "Sub-task 5: Compare the derived ranking of substances by increasing para-isomer weight fraction with the given multiple-choice options. "
        "Identify the correct sequence. Critically reflect on the final ranking, ensuring alignment with verified chemical principles and previous analysis. "
        "Use reflexion agent collaboration to catch residual inconsistencies before finalizing the answer."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
