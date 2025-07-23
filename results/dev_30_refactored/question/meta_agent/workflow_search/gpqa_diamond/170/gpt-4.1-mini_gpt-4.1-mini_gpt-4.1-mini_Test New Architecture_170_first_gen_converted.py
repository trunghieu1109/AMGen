async def forward_170(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                'Sub-task 1: Extract and summarize the given chemical information about the six substances and their substituents, including reaction conditions and assumptions. '
                'Input: taskInfo containing the question and choices.'
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results1, log1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results1['answer'])
        logs.append(log1)

        cot_agent_desc_2 = {
            'instruction': (
                'Sub-task 2: Analyze the electronic effects and directing properties of each substituent to predict regioselectivity trends for electrophilic bromination. '
                'Input: taskInfo and all previous thinking and answers from stage_0.subtask_1 iterations.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results2, log2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Estimate the relative weight fractions of the para-isomer formed for each substance based on substituent effects and known directing tendencies. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_2 and all previous iterations of stage_0.subtask_3.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_3', 'answer of previous stage_0.subtask_3']
        }
        results3, log3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

        cot_agent_desc_4 = {
            'instruction': (
                'Sub-task 4: Refine the para-isomer yield estimations by considering steric hindrance and other subtle factors influencing substitution patterns. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_3 and all previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
        }
        results4, log4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

        cot_agent_desc_5 = {
            'instruction': (
                'Sub-task 5: Consolidate and document the final refined para-isomer weight fractions for all substances, preparing data for comparison. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_4 and all previous iterations of stage_0.subtask_5.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4', 'thinking of previous stage_0.subtask_5', 'answer of previous stage_0.subtask_5']
        }
        results5, log5 = await self.cot(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_5
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results5['answer'])
        logs.append(log5)

    debate_desc_1 = {
        'instruction': (
            'Sub-task 1: Compare the consolidated para-isomer weight fractions across all substances to determine their relative order. '
            'Input: taskInfo and all thinking and answers from all iterations of stage_0.subtask_5.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Synthesize the comparison results and choose the most consistent ordering of substances by increasing para-isomer weight fraction.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_stage1_subtask1, log_stage1_subtask1 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=2
    )
    logs.append(log_stage1_subtask1)

    aggregate_desc_2 = {
        'instruction': (
            'Sub-task 2: Select the best candidate ordering of substances by increasing para-isomer weight fraction and justify the choice. '
            'Input: taskInfo and thinking and answer from stage_1.subtask_1.'
        ),
        'input': [taskInfo, results_stage1_subtask1['thinking'], results_stage1_subtask1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_stage1_subtask2, log_stage1_subtask2 = await self.aggregate(
        subtask_id='stage_1.subtask_2',
        aggregate_desc=aggregate_desc_2
    )
    logs.append(log_stage1_subtask2)

    final_answer = await self.make_final_answer(results_stage1_subtask2['thinking'], results_stage1_subtask2['answer'])
    return final_answer, logs
