async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the given chemical information: identify the six substances, their substituents, "
        "and the reaction conditions (electrophilic bromination with excess bromine, formation of only one monobromo derivative). "
        "Ensure clear understanding of the problem context to avoid oversimplification."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Clarify and define key terms and assumptions, especially the meaning of 'weight fraction of the para-isomer' "
        "(e.g., mass fraction vs. mole fraction) and critically evaluate how the assumption of a single monobromo derivative affects regioselectivity and product distribution. "
        "This subtask aims to prevent oversimplified interpretations that could skew later analysis."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for clarifying key terms and assumptions."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Analyze the electronic and steric effects of each substituent on the benzene ring to determine their directing influence (ortho/para or meta) "
        "and expected impact on para-isomer yield. Explicitly distinguish between electron-donating groups (EDGs), halogens, and electron-withdrawing groups (EWGs), "
        "and incorporate steric hindrance considerations. Avoid inconsistent or contradictory interpretations by grounding analysis in established chemical principles."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a reasoned consensus on substituent effects and their impact on para-isomer yield."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Perform a focused, quantitative comparison of the three meta-directing EWGs (–NO2, –COOH, –COOR) using empirical data such as Hammett σ_meta constants or known bromination regioselectivity ratios. "
        "Explicitly rank these substituents by their electron-withdrawing strength and expected para-isomer yield, with citations or approximate values to support the ranking."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide a quantitative ranking of meta-directing substituents by para-isomer yield."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Integrate the substituent electronic and steric effects (from subtask_3) with the quantitative meta-directing substituent ranking (from subtask_4) "
        "and the clarified assumptions (from subtask_2) to estimate the relative weight fractions of the para-isomer for each substance. "
        "Explicitly address the unique case of the halogen substituent (Cl) as a deactivating but ortho/para directing group, placing it correctly in the overall order. "
        "Resolve any remaining contradictions and produce a consistent, chemically sound para-isomer yield ordering."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions of integrating substituent effects and rankings to estimate para-isomer yields."
    )
    cot_reflect_desc5 = {
        "instruction": cot_reflect_instruction5,
        "critic_instruction": critic_instruction5,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Select and arrange the six substances in order of increasing weight fraction of the para-isomer yield based on the integrated analysis from subtask_5. "
        "Compare the resulting order with the given answer choices to identify the correct sequence. Use a structured consensus or weighted voting mechanism to explicitly resolve any residual conflicts or uncertainties before finalizing the answer."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Provide the final ordered sequence of substances by para-isomer weight fraction and identify the matching answer choice."
    )
    debate_desc6 = {
        "instruction": debate_instruction6,
        "final_decision_instruction": final_decision_instruction6,
        "input": [taskInfo, results5["thinking"], results5["answer"]],
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "temperature": 0.5
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6["thinking"], results6["answer"])
    return final_answer, logs
