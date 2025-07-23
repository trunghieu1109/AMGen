async def forward_165(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the quantitative expression for the pseudo-Goldstone boson mass squared M_{h_2}^2 "
        "from the given Lagrangian and VEV information, identifying the relevant mass scales and coefficients."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Construct intermediate representations of the radiative correction contributions to M_{h_2}^2, "
        "detailing which particles contribute positively or negatively and how the VEV combination (x^2 + v^2) factors into the formula."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Debate and synthesize the contributions and VEV role in the radiative correction formula for M_{h_2}^2."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Apply the transformation rules and theoretical knowledge of radiative corrections in extended Higgs sectors "
        "to derive the approximate formula for M_{h_2}^2, clarifying the role of each term and the overall normalization."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and finalize the derived approximate formula for M_{h_2}^2 with detailed term roles and normalization."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the four candidate formulas for M_{h_2}^2 against the derived theoretical expression "
        "and select the best candidate that correctly approximates the pseudo-Goldstone boson mass through radiative corrections."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best candidate formula for M_{h_2}^2 based on theoretical consistency and correctness."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_4.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
