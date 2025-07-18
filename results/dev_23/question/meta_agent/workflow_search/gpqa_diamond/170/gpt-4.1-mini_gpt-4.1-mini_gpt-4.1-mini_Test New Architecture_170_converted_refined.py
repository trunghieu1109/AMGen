async def forward_170(self, taskInfo):
    logs = []

    cot_reflect_instruction1 = (
        "Sub-task 1: Analyze and classify each substituent (CH3, COOC2H5, Cl, NO2, C2H5, COOH) on the benzene ring with a detailed, literature-backed evaluation of their electronic effects (electron-donating or withdrawing) and directing effects (ortho/para or meta directing). Explicitly address borderline or ambiguous substituents such as esters (COOC2H5) by considering resonance, inductive effects, and experimental data. Avoid lumping all EWGs together; differentiate their relative strengths and directing tendencies. Use references such as Hammett sigma values and known electrophilic substitution behavior."
    )
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    results1, log1 = await self.reflexion(
        subtask_id='subtask_1',
        reflect_desc=cot_reflect_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Collect and integrate quantitative data relevant to electrophilic bromination para-isomer yields for each substituent, including experimental para/ortho/meta product ratios, Hammett constants, and steric considerations. Use this data to support a nuanced, numeric estimation of para-isomer weight fractions for each substituted benzene."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    results2, log2 = await self.debate(
        subtask_id='subtask_2',
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Based on the refined substituent classification and quantitative data, predict and estimate the relative weight fractions of the para-bromo isomer formed for each substituted benzene. Provide explicit numeric or percentage estimates for each substituent, ensuring a strict increasing order without grouping equal yields for different substituents. Critically evaluate steric and electronic factors influencing regioselectivity, including exceptions such as halogens and esters."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': [
            'user query',
            'thinking of subtask 1', 'answer of subtask 1',
            'thinking of subtask 2', 'answer of subtask 2'
        ]
    }
    results3, log3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Arrange the six substances in a strict order of increasing para-isomer weight fraction yield according to the numeric predictions from subtask_3. Explicitly justify the ordering, especially among closely related substituents (NO2, COOH, COOC2H5), ensuring no ties or approximate equalities. Use a collaborative Reflexion pattern to refine and finalize the strict sequence."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': [
            'user query',
            'thinking of subtask 3', 'answer of subtask 3'
        ]
    }
    results4, log4 = await self.reflexion(
        subtask_id='subtask_4',
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Compare the predicted strict increasing order of para-isomer yields with the given multiple-choice options and select the correct sequence that matches the predicted order exactly. Provide reasoning for the selection and discuss why other options are inconsistent with the refined analysis."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': [
            'user query',
            'thinking of subtask 4', 'answer of subtask 4'
        ]
    }
    results5, log5 = await self.sc_cot(
        subtask_id='subtask_5',
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
