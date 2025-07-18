async def forward_193(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Enumerate all possible spin configurations (S1, S2, S3) where each spin is ±1, "
        "and calculate the energy E for each configuration using E = -J(S1S2 + S1S3 + S2S3)."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Group the spin configurations by their energy values to determine the degeneracy "
        "(number of configurations) corresponding to each distinct energy level, based on outputs from stage_0.subtask_1."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent grouping of energy levels and degeneracies.",
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Compute the partition function Z by summing over all energy levels: "
        "Z = sum(degeneracy * exp(-beta * E)), substituting the energies and degeneracies found in stage_1.subtask_1."
    )
    cot_sc_desc_1_2 = {
        "instruction": cot_sc_instruction_1_2,
        "final_decision_instruction": "Sub-task 2: Synthesize and choose the most consistent computed partition function Z.",
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc_1_2,
        n_repeat=self.max_sc
    )
    logs.append(log_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Compare the derived partition function expression from stage_1.subtask_2 "
        "with the given choices and select the correct one that matches the computed Z."
    )
    final_decision_instruction_2_1 = "Sub-task 1: Select the correct partition function expression matching the computed Z."
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1["thinking"], results_2_1["answer"])
    return final_answer, logs
