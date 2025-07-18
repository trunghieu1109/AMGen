async def forward_190(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze the first transformation: treatment of the starting material with sodium hydride followed by benzyl bromide, "
        "to determine the structural changes, especially the alkylation of the hydroxymethyl group, with context from the query."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the second transformation: reaction of product 1 with p-toluenesulfonyl hydrazide in catalytic HCl, "
        "to understand the formation of the tosyl hydrazone derivative and its structural implications, based on Sub-task 1 output."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the second transformation analysis."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Analyze the third transformation: treatment of product 2 with n-butyllithium at low temperature followed by aqueous ammonium chloride, "
        "to deduce chemical changes such as rearrangements or substitutions, based on Sub-task 2 output."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent answer for the third transformation analysis."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Analyze the fourth transformation: catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere, "
        "to determine which functional groups are reduced or removed and the resulting structural changes, based on Sub-task 3 output."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent answer for the fourth transformation analysis."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Integrate the structural changes from all four transformations to deduce the overall structural evolution "
        "from the starting material to product 4, including stereochemical and regiochemical considerations where applicable."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of the provided solutions of the integrated structural analysis from subtasks 1 to 4."
    )
    cot_reflect_desc5 = {
        "instruction": cot_reflect_instruction5,
        "critic_instruction": critic_instruction5,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "temperature": 0.0,
        "context_desc": [
            "user query",
            "thinking of subtask 1", "answer of subtask 1",
            "thinking of subtask 2", "answer of subtask 2",
            "thinking of subtask 3", "answer of subtask 3",
            "thinking of subtask 4", "answer of subtask 4"
        ]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Evaluate the four given product choices against the deduced structure of product 4, "
        "selecting the correct structure that matches the integrated analysis of the synthetic sequence."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Select the correct product structure based on the integrated analysis from Sub-task 5."
    )
    debate_desc6 = {
        "instruction": debate_instruction6,
        "final_decision_instruction": final_decision_instruction6,
        "input": [taskInfo, results5["thinking"], results5["answer"]],
        "context_desc": ["user query", "thinking of subtask 5", "answer of subtask 5"],
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
