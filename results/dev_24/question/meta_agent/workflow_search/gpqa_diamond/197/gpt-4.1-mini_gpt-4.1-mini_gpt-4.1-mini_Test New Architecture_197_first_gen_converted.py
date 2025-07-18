async def forward_197(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Analyze and classify the given chemical data: total cobalt concentration, thiocyanate concentration, "
        "and cumulative stability constants for Co(II)-SCN complexes. Identify the species involved and clarify assumptions about the blue dithiocyanato complex."
    )
    cot_agent_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Assess the equilibrium relationships and constraints governing the distribution of cobalt species in solution, "
        "including the role of cumulative stability constants and the effect of ligand concentration on complex formation."
    )
    cot_agent_desc_stage1_sub1 = {
        'instruction': cot_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_stage1_sub1
    )
    logs.append(log_stage1_sub1)

    cot_instruction_stage2_sub1 = (
        "Sub-task 1: Derive expressions for the concentrations of all cobalt species (free Co(II) and complexes Co(SCN)n) "
        "in terms of free Co(II) concentration, SCN- concentration, and stability constants."
    )
    cot_agent_desc_stage2_sub1 = {
        'instruction': cot_instruction_stage2_sub1,
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage2_sub1, log_stage2_sub1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_stage2_sub1
    )
    logs.append(log_stage2_sub1)

    cot_reflect_instruction_stage2_sub2 = (
        "Sub-task 2: Formulate the mass balance equation for total cobalt concentration and solve for the free Co(II) concentration "
        "using the derived expressions from Sub-task 1."
    )
    critic_instruction_stage2_sub2 = (
        "Please review and provide the limitations of provided solutions for this Sub-task 2: solving for free Co(II) concentration."
    )
    cot_reflect_desc_stage2_sub2 = {
        'instruction': cot_reflect_instruction_stage2_sub2,
        'critic_instruction': critic_instruction_stage2_sub2,
        'input': [taskInfo, results_stage2_sub1['thinking'], results_stage2_sub1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_stage2_sub2, log_stage2_sub2 = await self.reflexion(
        subtask_id="stage_2.subtask_2",
        reflect_desc=cot_reflect_desc_stage2_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub2)

    cot_instruction_stage3_sub1 = (
        "Sub-task 1: Calculate the concentration of the dithiocyanato complex Co(SCN)2 using the free Co(II) concentration and stability constants, "
        "then compute its percentage relative to total cobalt concentration."
    )
    cot_agent_desc_stage3_sub1 = {
        'instruction': cot_instruction_stage3_sub1,
        'input': [taskInfo, results_stage2_sub2['thinking'], results_stage2_sub2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_stage3_sub1, log_stage3_sub1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_stage3_sub1
    )
    logs.append(log_stage3_sub1)

    final_answer = await self.make_final_answer(results_stage3_sub1['thinking'], results_stage3_sub1['answer'])
    return final_answer, logs
