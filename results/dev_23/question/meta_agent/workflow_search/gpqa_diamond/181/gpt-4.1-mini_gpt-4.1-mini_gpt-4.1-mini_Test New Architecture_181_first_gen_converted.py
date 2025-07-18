async def forward_181(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Analyze and classify the defining attributes of the Mott-Gurney equation and the conditions described in the statements, "
        "including carrier type, trap presence, contact type, and current components."
    )
    cot_agent_desc_stage0 = {
        'instruction': cot_instruction_stage0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0, log_stage0 = await self.cot(
        subtask_id="stage0_subtask1",
        cot_agent_desc=cot_agent_desc_stage0
    )
    logs.append(log_stage0)

    cot_reflect_instruction_stage1 = (
        "Sub-task 1: Based on the output from Stage 0 Sub-task 1, assess the impact of trap presence, carrier type (single vs two-carrier), "
        "contact type (Ohmic vs Schottky), and current components (drift vs diffusion) on the validity of the Mott-Gurney equation."
    )
    critic_instruction_stage1 = (
        "Please review the assessment of the impact of trap presence, carrier type, contact type, and current components on the validity of the Mott-Gurney equation, "
        "and provide its limitations."
    )
    cot_reflect_desc_stage1 = {
        'instruction': cot_reflect_instruction_stage1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results_stage1, log_stage1 = await self.reflexion(
        subtask_id="stage1_subtask1",
        reflect_desc=cot_reflect_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Based on the outputs of Stage 0 Sub-task 1 and Stage 1 Sub-task 1, evaluate and prioritize the four given statements "
        "to determine which statement correctly describes the validity conditions of the Mott-Gurney equation."
    )
    debate_desc_stage2 = {
        'instruction': debate_instruction_stage2,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage1_subtask1", "answer of stage1_subtask1"],
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer'], results_stage1['thinking'], results_stage1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage2['thinking'], results_stage2['answer'])

    return final_answer, logs
