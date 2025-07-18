async def forward_196(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and interpret the IR and 1H NMR spectral data of Compound X, "
        "deducing the functional groups and substitution pattern with context from the given query."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc={
            'instruction': cot_instruction1,
            'final_decision_instruction': "Sub-task 1: Decide on the key functional groups and substitution pattern based on spectral data.",
            'input': [taskInfo],
            'context_desc': ["user query"],
            'temperature': 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the spectral analysis from Sub-task 1, analyze the reaction conditions involving red phosphorus and HI, "
        "and deduce the expected chemical transformation on Compound X."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and plausible chemical transformation outcome "
        "given the reaction conditions and spectral data analysis from Sub-task 1."
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

    debate_instruction3 = (
        "Sub-task 3: Evaluate the aliphatic proton splitting patterns in the 1H NMR spectrum to identify the nature of the alkyl substituent "
        "(sec-butyl, isobutyl, ethyl) on the aromatic ring, using the spectral data and analysis from Sub-task 1."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Decide on the most likely alkyl substituent based on the splitting patterns and spectral evidence."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Compare the deduced structure and reaction outcome from Sub-tasks 2 and 3 with the given product choices, "
        "and select the most plausible final product after reaction with red phosphorus and HI."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Decide on the best matching final product choice based on integrated spectral and reaction analysis."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Conclude and justify the identity of the final product based on the integrated spectral analysis and reaction mechanism evaluation "
        "from Sub-task 4."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide a final justified answer identifying the final product of the reaction."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
