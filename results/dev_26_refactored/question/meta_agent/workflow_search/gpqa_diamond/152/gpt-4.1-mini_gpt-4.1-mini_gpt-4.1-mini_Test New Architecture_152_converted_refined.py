async def forward_152(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Perform detailed structural mapping and carbon numbering of all reactants in reactions (A), (B), and (C). "
        "Explicitly label each carbon atom in nucleophiles and electrophiles, especially the α,β-unsaturated carbonyl compounds, to ensure correct identification of the β-carbon and the site of nucleophilic attack. "
        "This step aims to prevent misassignment of substitution positions in the products by establishing a clear, stepwise framework for bond formation."
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
        "Sub-task 2: Apply the Michael addition mechanism to each reaction using the mapped structures from Subtask 1. "
        "Predict the intermediate and final product structures by explicitly showing the new bond formation at the β-carbon, ensuring correct regiochemistry and substitution patterns. "
        "Avoid assumptions about tautomeric forms at this stage; focus on connectivity and substitution positions only."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the Michael addition product structures based on Subtask 1 analysis."
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
        "Sub-task 3: Analyze the tautomeric equilibria and protonation states of the predicted products from Subtask 2, "
        "especially for products B and C, under the given reaction conditions (e.g., basic aqueous media for reaction C, acidic or neutral for B). "
        "Use chemical stability data, conjugation effects, and literature precedence to determine the predominant tautomeric form (keto vs enol/hydroxy) and protonation state before finalizing product structures. "
        "This step addresses previous errors caused by incorrect tautomer assignments."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions regarding tautomeric and protonation state assignments in Subtask 3."
    )
    cot_reflect_desc3 = {
        "instruction": debate_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Conduct a rigorous sanity check and reflexion on the predicted products, focusing on verifying the regiochemistry, carbon numbering, and tautomeric assignments made in Subtasks 2 and 3. "
        "Challenge prior assumptions and cross-validate the product structures against IUPAC nomenclature rules and chemical reasoning to ensure consistency and correctness. "
        "This step is designed to catch subtle but critical errors before final answer selection."
    )
    critic_instruction4 = (
        "Please review and critique the structural assignments, tautomeric forms, and nomenclature consistency from previous subtasks."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Compare the fully analyzed and verified product structures from Subtask 4 with the multiple-choice options provided. "
        "Evaluate each choice based on correct substitution patterns, tautomeric forms, and nomenclature accuracy. "
        "Select the correct set of product identities (A, B, and C) that best matches the chemical reasoning and analysis performed in previous subtasks."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Select the correct multiple-choice answer for the Michael addition products based on comprehensive analysis."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
