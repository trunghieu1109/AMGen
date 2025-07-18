async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the starting material and the first reaction step (treatment with sodium hydride and benzyl bromide) "
        "to determine the structure of product 1. Focus on confirming the benzylation of the hydroxymethyl group to form a benzyl ether. "
        "Embed feedback to avoid assumptions without structural validation and ensure clear depiction of the protected intermediate."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze the second reaction step (treatment of product 1 with p-toluenesulfonyl hydrazide and catalytic HCl) "
        "to determine the structure of product 2. Focus on hydrazone formation at the ketone position, ensuring clear structural representation "
        "and avoiding ambiguity about regiochemistry and stereochemistry."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Analyze the third reaction step (treatment of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride) "
        "to determine the structure of product 3. Focus on the Shapiro reaction mechanism, explicitly illustrating the elimination of N2 and TsO–, "
        "formation of the vinyllithium intermediate, and generation of the exocyclic alkene at C-1. Avoid misinterpretation of the intermediate as an internal alkene. "
        "This subtask must produce a clear mechanistic rationale and structural depiction to prevent previous errors."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_2", "answer of subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_agent_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Conduct a mechanistic verification and reflection on the Shapiro reaction outcome from subtask_3. "
        "Agents must explicitly confirm or challenge the proposed exocyclic alkene intermediate and the vinyllithium mechanism. "
        "If disagreement arises, escalate to a Reflexion micro-task to reach consensus before proceeding. "
        "This step is critical to prevent propagation of errors from misinterpretation of the Shapiro reaction."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask_3", "answer of subtask_3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Analyze the fourth reaction step (stirring product 3 with Pd/C under hydrogen atmosphere) to determine the structure of product 4. "
        "Critically evaluate the reaction conditions to assess whether benzyl ether cleavage occurs or if the benzyl protecting group remains intact. "
        "Consider literature precedence, reaction time, temperature, catalyst loading, and selectivity of hydrogenation for alkenes versus benzyl ethers. "
        "Avoid assumptions about deprotection without evidence. Provide a detailed mechanistic rationale for the expected product structure."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_sc_instruction6 = (
        "Sub-task 6: Integrate the structural information and mechanistic insights from products 1 through 4 to select the correct final product structure from the given choices. "
        "Apply multi-criteria evaluation including functional group presence, protecting group status, and expected reaction outcomes. "
        "Explicitly verify all key assumptions made in previous steps, especially regarding the Shapiro reaction intermediate and benzyl ether stability. "
        "Use Self-Consistency Chain-of-Thought to ensure thorough reasoning and consensus on the final answer."
    )
    cot_sc_desc6 = {
        'instruction': cot_sc_instruction6,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_5", "answer of subtask_5"]
    }
    results6, log6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_sc_desc6,
        n_repeat=self.max_sc
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
