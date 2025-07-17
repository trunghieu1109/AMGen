async def forward_171(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the given quantitative information: excitation ratio (2), energy difference ΔE = 1.38×10^-23 J, "
        "and the assumption of LTE. Emphasize the importance of the Boltzmann constant k and its role in the problem to prevent omission in later steps."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_s1_st1, log_s1_st1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log_s1_st1)

    cot_sc_instruction2 = (
        "Sub-task 2: Apply the Boltzmann distribution to express the ratio of excited state populations in terms of ΔE, Boltzmann constant k, and temperatures T1 and T2. "
        "Explicitly write the formula ln(2) = (ΔE/k) * (1/T2 - 1/T1) without any simplification or omission of ΔE/k. "
        "Highlight the dimensional units of each term to prepare for consistency checks."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results_s1_st1.get('thinking', ''), results_s1_st1.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_s1_st2, log_s1_st2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log_s1_st2)

    cot_sc_instruction3 = (
        "Sub-task 1: Rearrange the expression from stage_1.subtask_2 to isolate a formula relating ln(2) to T1 and T2, explicitly carrying the ΔE/k factor through all algebraic manipulations. "
        "Perform a dimensional analysis and physical consistency check on the derived formula to ensure no terms are dropped or incorrectly simplified. "
        "Document the units and verify that the formula is dimensionally consistent and physically meaningful."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results_s1_st2.get('thinking', ''), results_s1_st2.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_s2_st1, log_s2_st1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log_s2_st1)

    debate_instruction1 = (
        "Sub-task 1: Compare the fully derived and dimensionally consistent equation with the four candidate equations. "
        "Carefully match the algebraic form including the ΔE/k factor and confirm which candidate (if any) corresponds exactly to the derived expression. "
        "Avoid mislabeling the choice letter by explicitly verifying the equivalence of the expressions including constants and variables."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'context': ["user query", results_s2_st1.get('thinking', ''), results_s2_st1.get('answer', '')],
        'input': [taskInfo, results_s2_st1.get('thinking', ''), results_s2_st1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_s3_st1, log_s3_st1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log_s3_st1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Perform a final verification step to confirm the correctness of the selected candidate equation and the assigned choice letter. "
        "Re-examine the dimensional consistency and physical interpretation of the matched candidate to ensure no conceptual or algebraic errors remain. "
        "This step aims to prevent previous mistakes of misassignment and omission of critical factors."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results_s3_st1.get('thinking', ''), results_s3_st1.get('answer', '')],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", results_s3_st1.get('thinking', ''), results_s3_st1.get('answer', '')]
    }
    results_s3_st2, log_s3_st2 = await self.reflexion(
        subtask_id="stage_3.subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log_s3_st2)

    final_answer = await self.make_final_answer(results_s3_st2.get('thinking', ''), results_s3_st2.get('answer', ''))
    return final_answer, logs
