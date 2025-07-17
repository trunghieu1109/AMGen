async def forward_150(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Analyze and classify the given system state vector and observable matrix operator, "
        "including checking normalization of the state vector and confirming the Hermitian property of the observable, "
        "with context from the provided query."
    )
    cot_agent_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    debate_instruction_stage1_sub1 = (
        "Sub-task 1: Find the eigenvalues and eigenvectors of the observable matrix operator P, "
        "and identify the eigenspace corresponding to the eigenvalue 0, based on the analysis from stage_0.subtask_1."
    )
    debate_desc_stage1_sub1 = {
        'instruction': debate_instruction_stage1_sub1,
        'context': ["user query", results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    cot_instruction_stage1_sub2 = (
        "Sub-task 2: Normalize the given state vector to ensure it is a valid quantum state for probability calculations, "
        "based on the analysis from stage_0.subtask_1."
    )
    cot_agent_desc_stage1_sub2 = {
        'instruction': cot_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.0,
        'context': ["user query", results_stage0_sub1['thinking'], results_stage0_sub1['answer']]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_stage1_sub2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Project the normalized state vector onto the eigenspace corresponding to eigenvalue 0 "
        "and compute the probability of measuring 0 by calculating the squared magnitude of this projection, "
        "based on outputs from stage_1.subtask_1 and stage_1.subtask_2."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'context': [
            "user query",
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        'input': [
            taskInfo,
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
