async def forward_193(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Enumerate all possible spin configurations (S1, S2, S3) where each spin is ±1, "
        "with context from the given problem about the three-spin system and energy expression."
    )
    cot_agent_desc_0_0 = {
        'instruction': cot_instruction_0_0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_0, log_0_0 = await self.cot(subtask_id='stage_0.subtask_0', cot_agent_desc=cot_agent_desc_0_0)
    logs.append(log_0_0)

    cot_instruction_0_1 = (
        "Sub-task 1: Calculate the energy E = -J(S1S2 + S1S3 + S2S3) for each spin configuration enumerated in Sub-task 0, "
        "using the spin values and given formula."
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo, results_0_0['thinking'], results_0_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_0', 'answer of stage_0.subtask_0']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Compute the Boltzmann factor e^{-βE} for each configuration using the energies calculated in Sub-task 1, "
        "considering β = 1/(kT)."
    )
    cot_agent_desc_0_2 = {
        'instruction': cot_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Group configurations by their energy values and count the multiplicities of each energy level, "
        "using the Boltzmann factors computed in Sub-task 2."
    )
    cot_agent_desc_0_3 = {
        'instruction': cot_instruction_0_3,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)

    aggregate_instruction_1_0 = (
        "Sub-task 0: Sum the Boltzmann factors of configurations grouped by energy to form partial sums for the partition function, "
        "based on the grouping from stage_0.subtask_3."
    )
    aggregate_desc_1_0 = {
        'instruction': aggregate_instruction_1_0,
        'input': [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_0.subtask_3']
    }
    results_1_0, log_1_0 = await self.aggregate(subtask_id='stage_1.subtask_0', aggregate_desc=aggregate_desc_1_0)
    logs.append(log_1_0)

    cot_instruction_1_1 = (
        "Sub-task 1: Express the partition function Z as a sum of exponentials with coefficients reflecting multiplicities, "
        "using the partial sums from Sub-task 0 of stage 1."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_0', 'answer of stage_1.subtask_0']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_instruction_2_0 = (
        "Sub-task 0: Compare the derived partition function expression with the given candidate choices, "
        "analyzing their structure and coefficients for consistency."
    )
    cot_agent_desc_2_0 = {
        'instruction': cot_instruction_2_0,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_0, log_2_0 = await self.cot(subtask_id='stage_2.subtask_0', cot_agent_desc=cot_agent_desc_2_0)
    logs.append(log_2_0)

    debate_instruction_2_1 = (
        "Sub-task 1: Validate the correctness of each candidate partition function by debating their consistency with energy multiplicities and Boltzmann factors, "
        "using the comparison from Sub-task 0 of stage 2."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Validate and debate the correctness of candidate partition functions to identify the consistent expression."
    )
    debate_desc_2_1 = {
        'instruction': debate_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        'context': ['user query', 'thinking of stage_2.subtask_0', 'answer of stage_2.subtask_0'],
        'temperature': 0.5
    }
    results_2_1, log_2_1 = await self.debate(subtask_id='stage_2.subtask_1', debate_desc=debate_desc_2_1, n_repeat=self.max_round)
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Select the candidate expression that matches the derived partition function based on the debate and validation results."
    )
    cot_agent_desc_2_2 = {
        'instruction': cot_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.cot(subtask_id='stage_2.subtask_2', cot_agent_desc=cot_agent_desc_2_2)
    logs.append(log_2_2)

    formatter_instruction_3_0 = (
        "Sub-task 0: Format the selected partition function expression into a clear, final answer statement, "
        "based on the selection from stage_2.subtask_2."
    )
    formatter_desc_3_0 = {
        'instruction': formatter_instruction_3_0,
        'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2'],
        'format': 'short and concise, without explanation'
    }
    results_3_0, log_3_0 = await self.specific_format(subtask_id='stage_3.subtask_0', formatter_desc=formatter_desc_3_0)
    logs.append(log_3_0)

    review_instruction_3_1 = (
        "Sub-task 1: Summarize the reasoning steps and final conclusion for presentation, "
        "based on the formatted final answer from Sub-task 0 of stage 3."
    )
    review_desc_3_1 = {
        'instruction': review_instruction_3_1,
        'input': [taskInfo, results_3_0['thinking'], results_3_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3.subtask_0', 'answer of stage_3.subtask_0']
    }
    results_3_1, log_3_1 = await self.review(subtask_id='stage_3.subtask_1', review_desc=review_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
