async def forward_152(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given chemical information, including reactants, reagents, reaction conditions, "
        "definitions related to Michael addition reactions, and the multiple-choice options. Ensure comprehensive capture of all relevant data to support downstream reasoning."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze and classify the chemical components and reaction conditions to understand their roles and relationships in the Michael addition context. "
        "Explicitly infer or clarify the identity and structure of the unknown compound C based on given information and choices, as this is critical for accurate product prediction."
    )
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Derive the initial expected major products of the three Michael addition reactions by applying organic chemistry principles such as nucleophilic attack, enolate formation, resonance stabilization, and regiochemistry. "
        "Focus on mechanistic steps leading to the primary addition products before considering tautomerism or protonation states."
    )
    critic_instruction3 = (
        "Sub-task 4: Evaluate the tautomeric equilibria, protonation states, and chemical plausibility of the initially derived products under the given reaction conditions (acidic, basic, aqueous, etc.). "
        "Explicitly determine which tautomeric form (enol or keto) or protonation state is thermodynamically favored for each product, especially for β-diketones and related compounds. "
        "Filter the valid scenarios that meet the conditions stated in the queries."
    )
    cot_reflect_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, convert the intermediate output into a final answer by comparing and matching the chemically plausible, tautomer-corrected final products with the multiple-choice options. "
        "Critically verify naming conventions, substituent positions, and tautomeric forms before selecting the correct choice."
    )
    debate_desc5 = {
        'instruction': debate_instruction_5,
        'context': ["user query", results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
