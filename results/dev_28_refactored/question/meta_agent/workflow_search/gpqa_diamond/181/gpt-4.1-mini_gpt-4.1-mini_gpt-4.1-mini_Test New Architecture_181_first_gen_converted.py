async def forward_181(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and categorize all relevant information from the query, including the Mott-Gurney equation parameters, "
        "the physical regime (SCLC), and the conditions described in each choice."
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

    cot_sc_instruction2_1 = (
        "Sub-task 1: Analyze the relationships and physical assumptions underlying the Mott-Gurney equation, "
        "such as trap-free conditions, single-carrier transport, contact types, and dominance of drift current over diffusion, "
        "based on the output from Stage 1 Subtask 1."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent analysis of the physical assumptions "
        "underlying the Mott-Gurney equation."
    )
    cot_sc_desc2_1 = {
        "instruction": cot_sc_instruction2_1,
        "final_decision_instruction": final_decision_instruction2_1,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    debate_instruction2_2 = (
        "Sub-task 2: Validate each of the four given statements against the theoretical and physical criteria "
        "for the Mott-Gurney equation's applicability, identifying which conditions are consistent or inconsistent with the model, "
        "based on outputs from Stage 1 Subtask 1 and Stage 2 Subtask 1."
    )
    final_decision_instruction2_2 = (
        "Sub-task 2: Provide a reasoned validation of each choice's correctness regarding the Mott-Gurney equation validity."
    )
    debate_desc2_2 = {
        "instruction": debate_instruction2_2,
        "final_decision_instruction": final_decision_instruction2_2,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2_1["thinking"], results2_1["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results2_2, log2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc2_2,
        n_repeat=self.max_round
    )
    logs.append(log2_2)

    debate_instruction3_1 = (
        "Sub-task 1: Select the best candidate statement that correctly describes the validity conditions "
        "of the Mott-Gurney equation based on the validation results from Stage 2 Subtask 2."
    )
    final_decision_instruction3_1 = (
        "Sub-task 1: Choose the most accurate and consistent statement describing the Mott-Gurney equation validity."
    )
    debate_desc3_1 = {
        "instruction": debate_instruction3_1,
        "final_decision_instruction": final_decision_instruction3_1,
        "input": [taskInfo, results2_2["thinking"], results2_2["answer"]],
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        "temperature": 0.5
    }
    results3_1, log3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3_1,
        n_repeat=self.max_round
    )
    logs.append(log3_1)

    final_answer = await self.make_final_answer(results3_1["thinking"], results3_1["answer"])
    return final_answer, logs
