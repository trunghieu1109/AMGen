async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the substituents on the benzene ring in terms of their electronic effects "
        "(activating/deactivating) and directing effects (ortho/para or meta) in electrophilic aromatic substitution, "
        "based on the given query and substances."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_0.subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the classification of substituents from Sub-task 1, summarize the influence of each substituent "
        "on the regioselectivity of bromination, focusing on the expected ratio of para- to ortho-isomers and the overall yield of the para-isomer."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate and compare the relative weight fractions of the para-isomer formed from each substituted benzene "
        "based on their substituent effects and steric considerations, using outputs from stage_0.subtask_2."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Arrange the six substances in order of increasing weight fraction of the para-isomer yield, "
        "using the evaluation from Sub-task 3."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Compare the arranged order from Sub-task 4 with the given multiple-choice options and identify the correct sequence."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="stage_1.subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
