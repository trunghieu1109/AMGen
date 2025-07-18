async def forward_193(self, taskInfo):
    logs = []

    cot_instruction0_1 = (
        "Sub-task 1: Enumerate all possible spin configurations (S1, S2, S3) where each spin is ±1, "
        "and prepare these configurations for energy calculation, with context from the user query."
    )
    cot_agent_desc0_1 = {
        "instruction": cot_instruction0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results0_1, log0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc0_1
    )
    logs.append(log0_1)

    cot_sc_instruction1_1 = (
        "Sub-task 1: Calculate the energy E = -J(S1S2 + S1S3 + S2S3) for each spin configuration enumerated in Stage 0, "
        "using the configurations from previous subtask and context from the user query."
    )
    final_decision_instruction1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent energy calculations for each configuration."
    )
    cot_sc_desc1_1 = {
        "instruction": cot_sc_instruction1_1,
        "final_decision_instruction": final_decision_instruction1_1,
        "input": [taskInfo, results0_1['thinking'], results0_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_sc_instruction1_2 = (
        "Sub-task 2: Group the configurations by their energy values and count the degeneracy (number of configurations) "
        "for each distinct energy level, based on the energy calculations from Sub-task 1."
    )
    final_decision_instruction1_2 = (
        "Sub-task 2: Synthesize and choose the most consistent grouping and degeneracy counts for energy levels."
    )
    cot_sc_desc1_2 = {
        "instruction": cot_sc_instruction1_2,
        "final_decision_instruction": final_decision_instruction1_2,
        "input": [taskInfo, results1_1['thinking'], results1_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    cot_sc_instruction1_3 = (
        "Sub-task 3: Compute the partition function Z by summing over all energy levels: "
        "Z = sum(degeneracy * exp(-beta * E)), using the grouped energies and degeneracies from Sub-task 2."
    )
    final_decision_instruction1_3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct partition function expression."
    )
    cot_sc_desc1_3 = {
        "instruction": cot_sc_instruction1_3,
        "final_decision_instruction": final_decision_instruction1_3,
        "input": [taskInfo, results1_2['thinking'], results1_2['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results1_3, log1_3 = await self.sc_cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_sc_desc1_3,
        n_repeat=self.max_sc
    )
    logs.append(log1_3)

    debate_instruction2_1 = (
        "Sub-task 1: Compare the computed partition function expression from Stage 1 Sub-task 3 "
        "with the given choices and select the correct one that matches the derived formula."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Select the correct partition function expression from the given choices based on the computed result."
    )
    debate_desc2_1 = {
        "instruction": debate_instruction2_1,
        "final_decision_instruction": final_decision_instruction2_1,
        "input": [taskInfo, results1_3['thinking'], results1_3['answer']],
        "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"],
        "temperature": 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])
    return final_answer, logs
