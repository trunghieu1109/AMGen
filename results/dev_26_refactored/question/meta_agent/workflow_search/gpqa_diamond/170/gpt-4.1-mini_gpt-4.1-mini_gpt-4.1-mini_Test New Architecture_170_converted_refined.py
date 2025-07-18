async def forward_170(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the chemical structures and substituents of the six substances, "
        "clarify the reaction conditions and assumptions (monobromo derivative, electrophilic substitution with excess bromine), "
        "and explicitly note the problem focus on weight fraction of para-isomer yield. "
        "Ensure clear understanding of the problem scope to avoid trivial or ambiguous interpretations."
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
        "Sub-task 2: Collect and analyze quantitative data on substituent electronic effects and directing influences, "
        "including Hammett sigma constants and empirical ortho/para ratios from authoritative chemical literature or databases. "
        "Explicitly address the previous failure of oversimplifying substituent effects by requiring numeric or literature-backed evidence, "
        "especially to correctly rank COOH vs. COOC2H5 and Cl vs. alkyl groups. "
        "Establish a reliable, quantitative basis for regioselectivity predictions."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and evidence-backed quantitative data for substituent effects and directing influences."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Evaluate steric and electronic factors influencing the relative yields of para- versus ortho-isomers for each substituent, "
        "integrating the quantitative data from Sub-task 2. Critically reconcile conflicting data and assumptions, "
        "using Reflexion to resolve discrepancies before finalizing estimates."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions regarding steric and electronic effects on regioselectivity, "
        "and suggest reconciliations or improvements."
    )
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="stage_1.subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Convert predicted mole fractions of para-isomers into weight fractions by incorporating the molar masses of substituents and brominated products. "
        "Ensure the final ranking reflects mass-based yields accurately, addressing prior oversight of treating mole fractions as weight fractions without adjustment."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and finalize the weight fraction conversions for para-isomer yields for all substances."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Stage 2 Sub-task 1: Integrate the quantitative substituent effect data, steric/electronic evaluations, and weight fraction conversions to produce a final, robust ranking of the six substances in order of increasing weight fraction of the para-bromo isomer yield. "
        "Critically evaluate all prior inputs, resolve any remaining conflicts through Debate, and ensure the ranking aligns with empirical and theoretical evidence, avoiding assumption-based errors."
    )
    final_decision_instruction5 = (
        "Stage 2 Sub-task 1: Produce the final ranking of substances by increasing para-isomer weight fraction yield, based on integrated evidence."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_instruction6 = (
        "Stage 2 Sub-task 2: Compare the derived final ranking with the provided multiple-choice options and select the correct order. "
        "Include a clear explanation of the rationale for the choice, referencing the quantitative and qualitative analyses from previous subtasks to justify the selection and prevent misinterpretation."
    )
    cot_agent_desc6 = {
        "instruction": cot_instruction6,
        "input": [taskInfo, results5["thinking"], results5["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results6, log6 = await self.sc_cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc6,
        n_repeat=self.max_sc
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6["thinking"], results6["answer"])
    return final_answer, logs
