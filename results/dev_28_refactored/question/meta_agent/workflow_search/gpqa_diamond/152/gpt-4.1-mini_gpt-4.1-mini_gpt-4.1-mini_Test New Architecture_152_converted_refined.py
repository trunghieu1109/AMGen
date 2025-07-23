async def forward_152(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and explicitly identify the structures of all reactants and reagents for reactions A, B, and C, "
        "including the nucleophiles, electrophiles, and reaction conditions. Clarify the identity of compound C as cyclohexane-1,3-dione (keto form) to avoid ambiguity. "
        "Provide detailed structural descriptions or drawings to establish a clear foundation for subsequent mechanistic reasoning."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Construct detailed mechanistic pathways for each Michael addition reaction (A, B, and C), "
        "including enolate formation, nucleophilic attack at the β-position, resonance stabilization, and post-addition transformations such as protonation, hydrolysis, or tautomerization steps. "
        "Specifically, model the hydrolysis of the enamine intermediate in reaction B to the corresponding ketone, and confirm the keto form of compound C in reaction C. "
        "Provide explicit intermediate and final product structures with correct regiochemistry and functional groups to avoid previous misassignments."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent mechanistic pathways and product structures for reactions A, B, and C based on Sub-task 1 analysis."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Perform a rigorous verification of the derived intermediate and final product structures from stage_1.subtask_2 by cross-checking them against mechanistic principles, nomenclature conventions, and the given multiple-choice options. "
        "Identify and resolve any inconsistencies in substitution positions, functional groups (e.g., hydroxy vs. oxo), and stereochemistry. "
        "This step aims to prevent propagation of errors by enforcing structural and mechanistic consistency before final answer selection."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Verify and finalize the correctness of product structures and mechanistic consistency for reactions A, B, and C."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the four provided multiple-choice product assignments against the verified product structures and mechanistic rationale from stage_2.subtask_1. "
        "Select the best matching candidate for each reaction (A, B, and C) with explicit justification based on structural, mechanistic, and nomenclature consistency. "
        "Address and clarify any subtle differences in naming or substitution patterns to ensure unambiguous final selection."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best multiple-choice answer for the Michael addition reactions A, B, and C with detailed justification."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Consolidate and refine the selected product assignments into a final, clear, and well-justified answer. "
        "Summarize the mechanistic reasoning, structural verifications, and rationale for the chosen multiple-choice option. "
        "Ensure the final output aligns fully with the reaction mechanisms, product structures, and expert feedback to avoid previous errors."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions and suggest improvements if any inconsistencies or ambiguities remain in the final answer."
    )
    cot_reflect_desc5 = {
        "instruction": cot_reflect_instruction5,
        "critic_instruction": critic_instruction5,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="stage_4.subtask_1",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
