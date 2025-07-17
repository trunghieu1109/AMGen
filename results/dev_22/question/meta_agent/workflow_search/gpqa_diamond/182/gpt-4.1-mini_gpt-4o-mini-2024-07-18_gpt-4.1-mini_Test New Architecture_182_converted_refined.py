async def forward_182(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the starting compound's structure, including the ring system, double bonds, and functional groups, "
        "and determine its initial index of hydrogen deficiency (IHD). Explicitly identify all unsaturation sources and ring contributions to avoid under- or over-counting IHD."
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
        "Sub-task 2: Critically enumerate and validate the known chemical transformations of each functional group (formyl, vinyl, carboxylic acid) "
        "and the ring unsaturation under the reaction conditions of red phosphorus and excess HI. Emphasize that red P/HI reduces oxygen-containing groups to alkyl iodides without hydrogenating C=C bonds or saturating rings. "
        "Use a Debate pattern to challenge and confirm mechanistic assumptions, preventing the previous error of assuming full hydrogenation and ring saturation."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    reflexion_instruction3 = (
        "Sub-task 3: Derive the most likely structure of the product after reaction, strictly based on the validated transformations from subtask_2 and the initial structure from subtask_1. "
        "Use Reflexion to iteratively refine the product structure, ensuring the ring remains intact and unsaturation is correctly represented, avoiding assumptions of ring loss or full saturation."
    )
    critic_instruction3 = (
        "Please review the derived product structure and provide its limitations or possible errors."
    )
    cot_reflect_desc3 = {
        'instruction': reflexion_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Compute the index of hydrogen deficiency (IHD) of the derived product structure by accurately counting rings and pi bonds, "
        "explicitly including the ring contribution and any remaining double bonds or unsaturation. Avoid previous mistakes of ignoring ring IHD contribution or miscounting due to incorrect product assumptions."
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

    debate_instruction5 = (
        "Sub-task 5: Select the correct IHD value from the given choices (0, 1, 3, 5) based on the computed IHD of the product. "
        "Use a Debate pattern to critically evaluate the computed IHD against the choices, ensuring consensus and preventing premature or incorrect selection."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
