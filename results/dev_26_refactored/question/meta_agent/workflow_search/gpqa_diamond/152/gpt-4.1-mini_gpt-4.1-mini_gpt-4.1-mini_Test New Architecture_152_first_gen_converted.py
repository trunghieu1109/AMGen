async def forward_152(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Apply the Michael addition reaction mechanism to each given reaction (A, B, C) to predict the intermediate and final product structures based on the reactants and reagents provided."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Combine and integrate the predicted product structures with the reaction conditions and reagents to refine the expected final products, considering tautomerism, protonation states, and stereochemistry where relevant."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent refined product structures for the Michael addition reactions."
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
        "Sub-task 3: Analyze and interpret the multiple-choice options by comparing the predicted and refined products with the given product names and structures, focusing on key functional groups, substitution patterns, and nomenclature correctness."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Determine the best matching multiple-choice option for the Michael addition products based on chemical reasoning and analysis."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Select the correct set of product identities (A, B, and C) from the multiple-choice options based on the chemical reasoning and analysis performed in previous subtasks."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Finalize the correct answer choice for the Michael addition reactions products."
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

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
