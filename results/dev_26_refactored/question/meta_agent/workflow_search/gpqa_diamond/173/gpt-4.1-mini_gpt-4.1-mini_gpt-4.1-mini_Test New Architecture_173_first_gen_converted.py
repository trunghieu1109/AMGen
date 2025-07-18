async def forward_173(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and transform the given physical parameters: initial mass M, rest-mass energies, "
        "fragment mass ratio, and total rest mass after fission into explicit numerical values and symbolic expressions suitable for further calculations, with context from the user query."
    )
    cot_agent_desc_stage0_sub1 = {
        "instruction": cot_instruction_stage0_sub1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    debate_instruction_stage1_sub1 = (
        "Sub-task 1: Apply conservation of momentum and energy to derive expressions for the velocities and momenta of the two fragments, "
        "using relativistic mechanics and the mass ratio constraints, based on the extracted parameters from stage_0.subtask_1."
    )
    final_decision_instruction_stage1_sub1 = (
        "Sub-task 1: Derive velocities and momenta of the two fragments using relativistic mechanics and mass ratio constraints."
    )
    debate_desc_stage1_sub1 = {
        "instruction": debate_instruction_stage1_sub1,
        "final_decision_instruction": final_decision_instruction_stage1_sub1,
        "input": [taskInfo, results_stage0_sub1["thinking"], results_stage0_sub1["answer"]],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    debate_instruction_stage1_sub2 = (
        "Sub-task 2: Calculate the relativistic kinetic energy T1 of the heavier fragment using the derived velocities and relativistic energy-momentum relations, "
        "based on the results from stage_1.subtask_1."
    )
    final_decision_instruction_stage1_sub2 = (
        "Sub-task 2: Calculate relativistic kinetic energy T1 of the heavier fragment."
    )
    debate_desc_stage1_sub2 = {
        "instruction": debate_instruction_stage1_sub2,
        "final_decision_instruction": final_decision_instruction_stage1_sub2,
        "input": [taskInfo, results_stage1_sub1["thinking"], results_stage1_sub1["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results_stage1_sub2, log_stage1_sub2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage1_sub3 = (
        "Sub-task 3: Calculate the classical (non-relativistic) kinetic energy approximation T1_classical of the heavier fragment using classical kinetic energy formula and the same momentum or velocity approximations, "
        "based on the results from stage_1.subtask_1."
    )
    final_decision_instruction_stage1_sub3 = (
        "Sub-task 3: Calculate classical kinetic energy approximation T1_classical of the heavier fragment."
    )
    debate_desc_stage1_sub3 = {
        "instruction": debate_instruction_stage1_sub3,
        "final_decision_instruction": final_decision_instruction_stage1_sub3,
        "input": [taskInfo, results_stage1_sub1["thinking"], results_stage1_sub1["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results_stage1_sub3, log_stage1_sub3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc_stage1_sub3,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub3)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Compute the absolute difference between the relativistic kinetic energy T1 and the classical kinetic energy approximation T1_classical for the heavier fragment, "
        "and select the closest answer choice from the given options, based on results from stage_1.subtask_2 and stage_1.subtask_3."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Compute the difference between relativistic and classical kinetic energies and select the closest answer choice."
    )
    debate_desc_stage2_sub1 = {
        "instruction": debate_instruction_stage2_sub1,
        "final_decision_instruction": final_decision_instruction_stage2_sub1,
        "input": [taskInfo, results_stage1_sub2["thinking"], results_stage1_sub2["answer"], results_stage1_sub3["thinking"], results_stage1_sub3["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"],
        "temperature": 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1["thinking"], results_stage2_sub1["answer"])
    return final_answer, logs
