async def forward_160(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_1.subtask_1': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1 (stage_0.subtask_1): Extract and summarize all given information from the query relevant to the mean free path problem. "
            "Input: [taskInfo]"
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2 (stage_0.subtask_2): Identify and categorize key physical parameters and variables affecting the mean free path (λ1 and λ2). "
            "Input: [taskInfo] + all previous thinking and answers from stage_0.subtask_1 iterations"
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_1']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_1_1 = (
            "Sub-task 1 (stage_1.subtask_1): Analyze the relationship between electron beam initiation and changes in mean free path, considering electron-gas scattering. "
            "Input: [taskInfo] + all previous thinking and answers from stage_0.subtask_2 iterations"
        )
        cot_agent_desc_1_1 = {
            'instruction': cot_instruction_1_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id='stage_1.subtask_1',
            cot_agent_desc=cot_agent_desc_1_1
        )
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])
        logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1 (stage_2.subtask_1): Derive the conclusion about λ2 relative to λ1 based on the analysis of scattering effects and vacuum conditions. "
        "Input: [taskInfo] + all thinking and answers from stage_0.subtask_2 and stage_1.subtask_1 iterations"
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_1.subtask_1']['answer'] + loop_results['stage_1.subtask_1']['thinking'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_agent_instruction_3_1 = (
        "Sub-task 1 (stage_3.subtask_1): Format the final conclusion clearly, selecting the correct choice about λ2 from the given options. "
        "Input: [taskInfo] + thinking and answer from stage_2.subtask_1"
    )
    cot_agent_desc_3_1 = {
        'instruction': cot_agent_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id='stage_3.subtask_1',
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
