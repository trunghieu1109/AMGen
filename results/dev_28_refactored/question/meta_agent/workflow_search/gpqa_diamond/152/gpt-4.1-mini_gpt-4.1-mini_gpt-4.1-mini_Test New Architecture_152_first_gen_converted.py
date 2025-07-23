async def forward_152(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Construct intermediate representations of the reactants, reagents, and reaction conditions for each Michael addition reaction (A, B, and C), "
        "including identification of nucleophiles, electrophiles, and expected enolate intermediates, with context from the given query."
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
        "Sub-task 2: Based on the intermediate representations from Sub-task 1, derive the major final products of each Michael addition reaction (A, B, and C) "
        "by applying Michael addition mechanisms, considering resonance stabilization, regiochemistry, and subsequent protonation or hydrolysis steps."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the major final products of the Michael addition reactions."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the four given multiple-choice product assignments against the derived products from Sub-task 2 to select the best matching candidate for reactions A, B, and C."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the best matching multiple-choice answer for the Michael addition reactions based on the derived products."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Refine and consolidate the selected product assignments from Sub-task 3 into a final, clear, and justified answer "
        "that aligns with the reaction mechanisms and product structures."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of the selected product assignments and suggest improvements if any."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="stage_4.subtask_1",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
