async def forward_170(self, taskInfo):
    logs = []

    cot_instruction_0_0 = "Sub-task 0: Extract and summarize the given chemical information about the six substances and their substituents, focusing on their structure and substituent types, with context from the query."
    cot_agent_desc_0_0 = {
        'instruction': cot_instruction_0_0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_0, log_0_0 = await self.cot(subtask_id='stage_0.subtask_0', cot_agent_desc=cot_agent_desc_0_0)
    logs.append(log_0_0)

    cot_instruction_0_1 = "Sub-task 1: Analyze the electronic effects of substituents on electrophilic substitution regioselectivity, focusing on para-isomer formation, based on the chemical information extracted in Sub-task 0."
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo, results_0_0['thinking'], results_0_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_0', 'answer of stage_0.subtask_0']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_instruction_0_2 = "Sub-task 2: Document assumptions and clarify ambiguous terms such as 'weight fraction of para-isomer yield' and reaction conditions, based on the analysis from Sub-task 1."
    cot_agent_desc_0_2 = {
        'instruction': cot_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_instruction_1_0 = "Sub-task 0: Combine the summarized chemical data and substituent effect analysis into a consolidated framework for ranking substances, using outputs from stage_0.subtask_2."
    cot_agent_desc_1_0 = {
        'instruction': cot_instruction_1_0,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_0, log_1_0 = await self.cot(subtask_id='stage_1.subtask_0', cot_agent_desc=cot_agent_desc_1_0)
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = "Sub-task 1: Apply evaluation criteria to estimate relative para-isomer yields based on substituent directing effects and steric factors, using the consolidated framework from Sub-task 0."
    cot_agent_desc_1_1 = {
        'instruction': cot_agent_instruction_1_1,
        'input': [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_0', 'answer of stage_1.subtask_0']
    }
    results_1_1, log_1_1 = await self.answer_generate(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    aggregate_instruction_2_0 = "Sub-task 0: Validate the consolidated ranking framework against known chemical principles and literature data if available, based on the estimated para-isomer yields from stage_1.subtask_1."
    aggregate_desc_2_0 = {
        'instruction': aggregate_instruction_2_0,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_1.subtask_1']
    }
    results_2_0, log_2_0 = await self.aggregate(subtask_id='stage_2.subtask_0', aggregate_desc=aggregate_desc_2_0)
    logs.append(log_2_0)

    cot_instruction_2_1 = "Sub-task 1: Select the most plausible order of substances by increasing para-isomer weight fraction from the given choices, based on validation results from Sub-task 0."
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_0', 'answer of stage_2.subtask_0']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    aggregate_instruction_2_2 = "Sub-task 2: Evaluate the validity of the selected order and prepare justification for the choice, using outputs from Sub-task 1."
    aggregate_desc_2_2 = {
        'instruction': aggregate_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.aggregate(subtask_id='stage_2.subtask_2', aggregate_desc=aggregate_desc_2_2)
    logs.append(log_2_2)

    formatter_instruction_3_0 = "Sub-task 0: Consolidate the selected order and its justification into a clear, formatted final answer, based on evaluation from stage_2.subtask_2."
    formatter_desc_3_0 = {
        'instruction': formatter_instruction_3_0,
        'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2'],
        'format': 'short and concise, without explanation'
    }
    results_3_0, log_3_0 = await self.specific_format(subtask_id='stage_3.subtask_0', formatter_desc=formatter_desc_3_0)
    logs.append(log_3_0)

    review_instruction_3_1 = "Sub-task 1: Review the final output for coherence, completeness, and alignment with the query requirements, based on the formatted answer from Sub-task 0."
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
