async def forward_150(self, taskInfo):
    logs = []

    stage0_results = {}

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and normalize the given state vector to ensure correct probability interpretation. "
        "Given the state vector (-1, 2, 1), normalize it and provide the normalized vector."
    )
    cot_agent_desc_0_0 = {
        'instruction': cot_instruction_0_0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_0, log_0_0 = await self.cot(subtask_id='stage_0.subtask_0', cot_agent_desc=cot_agent_desc_0_0)
    logs.append(log_0_0)
    stage0_results['subtask_0'] = results_0_0

    cot_instruction_0_1 = (
        "Sub-task 1: Compute the eigenvalues and eigenvectors of the observable matrix P. "
        "Matrix P is given as [[0, 1/sqrt(2), 0], [1/sqrt(2), 0, 1/sqrt(2)], [0, 1/sqrt(2), 0]]. "
        "Calculate all eigenvalues and corresponding eigenvectors."
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)
    stage0_results['subtask_1'] = results_0_1

    cot_instruction_0_2 = (
        "Sub-task 2: Identify the eigenvectors corresponding to the eigenvalue 0 to form the projection subspace. "
        "Use the eigenvalues and eigenvectors computed previously to find those associated with eigenvalue 0."
    )
    cot_agent_desc_0_2 = {
        'instruction': cot_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)
    stage0_results['subtask_2'] = results_0_2

    cot_instruction_0_3 = (
        "Sub-task 3: Construct the projection operator onto the eigenspace associated with eigenvalue 0 using the identified eigenvectors. "
        "Form the projection matrix that projects any vector onto the zero-eigenvalue eigenspace."
    )
    cot_agent_desc_0_3 = {
        'instruction': cot_instruction_0_3,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2']
    }
    results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)
    stage0_results['subtask_3'] = results_0_3

    cot_instruction_0_4 = (
        "Sub-task 4: Project the normalized state vector onto the zero-eigenvalue eigenspace using the projection operator. "
        "Use the normalized state vector and the projection operator to find the projected vector."
    )
    cot_agent_desc_0_4 = {
        'instruction': cot_instruction_0_4,
        'input': [taskInfo, results_0_0['thinking'], results_0_0['answer'], results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 0', 'answer of subtask 0', 'thinking of subtask 3', 'answer of subtask 3']
    }
    results_0_4, log_0_4 = await self.cot(subtask_id='stage_0.subtask_4', cot_agent_desc=cot_agent_desc_0_4)
    logs.append(log_0_4)
    stage0_results['subtask_4'] = results_0_4

    cot_instruction_0_5 = (
        "Sub-task 5: Calculate the squared magnitude of the projected vector to obtain the probability of measuring eigenvalue 0. "
        "Compute the inner product of the projected vector with itself to get the probability."
    )
    cot_agent_desc_0_5 = {
        'instruction': cot_instruction_0_5,
        'input': [taskInfo, results_0_4['thinking'], results_0_4['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4']
    }
    results_0_5, log_0_5 = await self.cot(subtask_id='stage_0.subtask_5', cot_agent_desc=cot_agent_desc_0_5)
    logs.append(log_0_5)
    stage0_results['subtask_5'] = results_0_5

    stage1_results = {}

    ag_instruction_1_0 = (
        "Sub-task 0: Simplify and consolidate the calculated probability expression to a final numeric or symbolic form. "
        "Use the probability calculated previously to express it in simplest form."
    )
    ag_desc_1_0 = {
        'instruction': ag_instruction_1_0,
        'input': [taskInfo, results_0_5['thinking'], results_0_5['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_0, log_1_0 = await self.answer_generate(subtask_id='stage_1.subtask_0', cot_agent_desc=ag_desc_1_0)
    logs.append(log_1_0)
    stage1_results['subtask_0'] = results_1_0

    ag_instruction_1_1 = (
        "Sub-task 1: Compare the simplified probability with the given answer choices and select the best matching candidate. "
        "Match the simplified probability to one of the provided answer choices: 1/3, 2/3, sqrt(2/3), or 1."
    )
    ag_desc_1_1 = {
        'instruction': ag_instruction_1_1,
        'input': [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_0', 'answer of stage_1.subtask_0']
    }
    results_1_1, log_1_1 = await self.answer_generate(subtask_id='stage_1.subtask_1', cot_agent_desc=ag_desc_1_1)
    logs.append(log_1_1)
    stage1_results['subtask_1'] = results_1_1

    stage2_results = {}

    cot_instruction_2_0 = (
        "Sub-task 0: Apply any necessary mathematical transformations or verifications to confirm the correctness of the selected answer. "
        "Verify the selected answer's correctness using quantum mechanics principles and matrix operations."
    )
    cot_agent_desc_2_0 = {
        'instruction': cot_instruction_2_0,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_0, log_2_0 = await self.cot(subtask_id='stage_2.subtask_0', cot_agent_desc=cot_agent_desc_2_0)
    logs.append(log_2_0)
    stage2_results['subtask_0'] = results_2_0

    ag_instruction_2_0 = (
        "Sub-task 0 (AnswerGenerate): Confirm the correctness of the verified answer and finalize the output."
    )
    ag_desc_2_0 = {
        'instruction': ag_instruction_2_0,
        'input': [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_0', 'answer of stage_2.subtask_0']
    }
    results_2_0_ag, log_2_0_ag = await self.answer_generate(subtask_id='stage_2.subtask_0_answer', cot_agent_desc=ag_desc_2_0)
    logs.append(log_2_0_ag)
    stage2_results['subtask_0_answer'] = results_2_0_ag

    stage3_results = {}

    review_instruction_3_0 = (
        "Sub-task 0: Evaluate the final answer for correctness, consistency with quantum mechanics principles, "
        "and alignment with problem requirements."
    )
    review_desc_3_0 = {
        'instruction': review_instruction_3_0,
        'input': [taskInfo, results_2_0_ag['thinking'], results_2_0_ag['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_0_answer', 'answer of stage_2.subtask_0_answer']
    }
    results_3_0, log_3_0 = await self.review(subtask_id='stage_3.subtask_0', review_desc=review_desc_3_0)
    logs.append(log_3_0)
    stage3_results['subtask_0'] = results_3_0

    final_answer = await self.make_final_answer(results_3_0['thinking'], results_3_0['answer'])

    return final_answer, logs
