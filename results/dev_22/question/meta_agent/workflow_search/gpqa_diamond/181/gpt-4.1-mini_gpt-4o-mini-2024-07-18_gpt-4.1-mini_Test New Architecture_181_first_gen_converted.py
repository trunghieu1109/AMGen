async def forward_181(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the defining attributes of the Mott-Gurney equation and the physical conditions "
        "(trap-free, single-carrier, contact types, diffusion vs drift current) with context from the user query."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log0 = await self.cot(
        subtask_id="stage0_subtask1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log0)

    cot_reflect_instruction2 = (
        "Sub-task 1: Assess the impact of each physical condition (trap states, carrier type, contact type, diffusion/drift current) "
        "on the validity of the Mott-Gurney equation, based on the output from Stage 0 Sub-task 1."
    )
    critic_instruction2 = (
        "Please review the assessment of the impact of physical conditions on the validity of the Mott-Gurney equation and provide its limitations."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results_stage1_sub1, log1 = await self.reflexion(
        subtask_id="stage1_subtask1",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log1)

    debate_instruction3 = (
        "Sub-task 1: Evaluate the four given statements against the analyzed conditions and assessed impacts "
        "to determine which statement correctly describes the validity of the Mott-Gurney equation, based on outputs from Stage 0 Sub-task 1 and Stage 1 Sub-task 1."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'input': [taskInfo, results_stage0_sub1, results_stage1_sub1],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub1, log2 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log2)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
