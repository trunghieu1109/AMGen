async def forward_190(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given chemical information, including starting material structure, reagents, reaction conditions, "
        "and possible mechanistic implications for each step, based on the provided query."
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
        "Sub-task 2: Based on the output from Sub-task 1, analyze the relationships and transformations between the starting material and each intermediate product (1 to 4), "
        "focusing on functional group changes, reaction mechanisms, and stereochemical considerations."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and comprehensive analysis for the transformations and relationships between intermediates."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Derive the detailed structures of intermediates 1, 2, and 3 by applying the transformations identified in Stage 1 and 2, "
        "including the site of alkylation, hydrazone formation, lithiation, and protonation steps."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and select the most consistent detailed structures for intermediates 1, 2, and 3."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Apply the catalytic hydrogenation step to intermediate 3 to deduce the structure of product 4, "
        "considering the reduction of double bonds, removal of protecting groups, and overall changes in substitution."
    )
    cot_agent_desc4 = {
        "instruction": cot_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Compare the deduced structure of product 4 with the provided multiple-choice options and select the correct structure based on the chemical transformations and reasoning."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Select the correct structure of product 4 from the given choices based on the analysis and transformations."
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
