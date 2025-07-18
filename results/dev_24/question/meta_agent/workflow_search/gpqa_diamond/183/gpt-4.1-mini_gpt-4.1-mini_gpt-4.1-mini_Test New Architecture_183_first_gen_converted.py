async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the role and effect of each reagent/step in the given sequences on benzene, "
        "focusing on the type of transformation (alkylation, nitration, reduction, diazotization, substitution, sulfonation, hydrolysis) "
        "and the expected regioselectivity and directing effects. Use the provided sequences and target molecule details."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, map the positional outcomes of each step in the sequences on the benzene ring, "
        "considering directing effects of substituents introduced at each stage, to predict intermediate substitution patterns and verify if they align with the target substitution pattern (2-tert-butyl, 1-ethoxy, 3-nitro)."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent positional mapping for each sequence, "
        "given all the above thinking and answers, to verify alignment with the target molecule."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
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
        "Sub-task 3: Evaluate and compare the four given sequences based on the predicted intermediates, regioselectivity, "
        "and feasibility of each step to determine which sequence most likely leads to the high-yield synthesis of the target compound."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a reasoned evaluation and ranking of the sequences based on the analysis."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Identify potential side reactions, yield-limiting steps, or regioselectivity issues in each sequence "
        "and assess their impact on the overall synthetic route efficiency and product purity."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide a detailed critique of each sequence highlighting possible problems and their consequences."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_1.subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Select and justify the optimal sequence of reactions from the given options that leads to the high-yield synthesis "
        "of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene, based on the evaluations and critiques from previous subtasks."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final recommended sequence and justification."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_1.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
