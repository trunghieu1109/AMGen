async def forward_193(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Enumerate all 8 possible spin configurations (S1, S2, S3) with each spin ±1 and compute the energy E = -J(S1S2 + S1S3 + S2S3) for each configuration, "
        "explaining the calculation step-by-step with context from the user query."
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
        "Sub-task 2: Based on the energies computed in Sub-task 1, group the 8 configurations by their energy values, "
        "identify distinct energy levels, and count the degeneracy (number of configurations) for each energy level, "
        "with detailed reasoning and self-consistency checks."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent grouping of energy levels and degeneracies for the system."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Construct the partition function Z as the sum over energy levels of the form Z = Σ (degeneracy × e^{-β E}), "
        "substituting the energies and degeneracies found in Sub-task 2, with detailed step-by-step reasoning."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and confirm the correct expression for the partition function Z based on previous results."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Compare the derived partition function expression from Sub-task 3 with the four given candidate expressions, "
        "debate their correctness, and select the one that matches exactly in terms of coefficients and exponents."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the correct partition function expression from the given choices based on the debate."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_4.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
