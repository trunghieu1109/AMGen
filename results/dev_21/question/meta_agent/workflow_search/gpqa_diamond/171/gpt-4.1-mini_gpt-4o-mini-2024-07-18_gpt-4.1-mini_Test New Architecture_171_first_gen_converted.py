async def forward_171(self, taskInfo):
    logs = []

    cot_instruction0_1 = "Sub-task 1: Extract and summarize the given quantitative information: excitation ratio (2), energy difference (ΔE = 1.38×10^-23 J), and the assumption of LTE from the user query."
    cot_agent_desc0_1 = {
        'instruction': cot_instruction0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0_1, log0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc0_1
    )
    logs.append(log0_1)

    cot_sc_instruction1_1 = "Sub-task 1: Apply the Boltzmann distribution to express the ratio of excited state populations in terms of ΔE, Boltzmann constant k, and temperatures T1 and T2, based on the summary from stage_0.subtask_1."
    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_sc_instruction1_2 = "Sub-task 2: Take the natural logarithm of the excitation ratio and rearrange the expression to isolate a formula relating ln(2) to T1 and T2, using outputs from stage_0.subtask_1 and stage_1.subtask_1."
    cot_sc_desc1_2 = {
        'instruction': cot_sc_instruction1_2,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    cot_sc_instruction2_1 = "Sub-task 1: Derive the explicit equation relating ln(2) to T1 and T2 based on the Boltzmann relation and given ΔE/k, simplifying to a form comparable to the candidate equations, using output from stage_1.subtask_2."
    cot_sc_desc2_1 = {
        'instruction': cot_sc_instruction2_1,
        'input': [taskInfo, results1_2['thinking'], results1_2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    debate_instruction3_1 = "Sub-task 1: Compare the derived equation from stage_2.subtask_1 with the four candidate equations and select the correct one that matches the derived relationship."
    debate_desc3_1 = {
        'instruction': debate_instruction3_1,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        'input': [taskInfo, results2_1['thinking'], results2_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3_1, log3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3_1,
        n_repeat=self.max_round
    )
    logs.append(log3_1)

    final_answer = await self.make_final_answer(results3_1['thinking'], results3_1['answer'])
    return final_answer, logs
