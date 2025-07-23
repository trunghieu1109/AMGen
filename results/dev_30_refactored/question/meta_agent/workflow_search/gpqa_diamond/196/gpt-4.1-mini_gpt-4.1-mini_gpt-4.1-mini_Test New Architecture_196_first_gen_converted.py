async def forward_196(self, taskInfo):
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
                "Sub-task 1: Analyze the IR spectral data to identify functional groups present in Compound X. "
                "Input content: taskInfo containing question and choices."
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

        cot_reflect_desc_2 = {
            'instruction': (
                "Sub-task 2: Analyze the 1H NMR spectral data to deduce structural features and substituent patterns in Compound X. "
                "Input content: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_0.subtask_2."
            ),
            'critic_instruction': (
                "Please review and provide the limitations of provided solutions of Sub-task 2: NMR analysis."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1',
                'thinking of previous stage_0.subtask_2 iterations', 'answer of previous stage_0.subtask_2 iterations'
            ]
        }
        results2, log2 = await self.reflexion(
            subtask_id='stage_0.subtask_2',
            reflect_desc=cot_reflect_desc_2,
            n_repeat=2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_reflect_desc_3 = {
            'instruction': (
                "Sub-task 3: Interpret the reaction conditions (red phosphorus and HI) to predict possible transformations of Compound X. "
                "Input content: results (thinking and answer) from stage_0.subtask_1, stage_0.subtask_2, and all previous iterations of stage_0.subtask_3."
            ),
            'critic_instruction': (
                "Please review and provide the limitations of provided solutions of Sub-task 3: Reaction interpretation."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1',
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of previous stage_0.subtask_3 iterations', 'answer of previous stage_0.subtask_3 iterations'
            ]
        }
        results3, log3 = await self.reflexion(
            subtask_id='stage_0.subtask_3',
            reflect_desc=cot_reflect_desc_3,
            n_repeat=2
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

        cot_reflect_desc_4 = {
            'instruction': (
                "Sub-task 4: Integrate spectral analysis and reaction interpretation to propose an initial structure for the final product. "
                "Input content: results (thinking and answer) from stage_0.subtask_2, stage_0.subtask_3, and all previous iterations of stage_0.subtask_4."
            ),
            'critic_instruction': (
                "Please review and provide the limitations of provided solutions of Sub-task 4: Integration and proposal."
            ),
            'input': loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3',
                'thinking of previous stage_0.subtask_4 iterations', 'answer of previous stage_0.subtask_4 iterations'
            ]
        }
        results4, log4 = await self.reflexion(
            subtask_id='stage_0.subtask_4',
            reflect_desc=cot_reflect_desc_4,
            n_repeat=2
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

        cot_reflect_desc_5 = {
            'instruction': (
                "Sub-task 5: Refine the proposed final product structure by consolidating all evidence and addressing any inconsistencies. "
                "Input content: results (thinking and answer) from stage_0.subtask_4 and all previous iterations of stage_0.subtask_5."
            ),
            'critic_instruction': (
                "Please review and provide the limitations of provided solutions of Sub-task 5: Refinement."
            ),
            'input': loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4',
                'thinking of previous stage_0.subtask_5 iterations', 'answer of previous stage_0.subtask_5 iterations'
            ]
        }
        results5, log5 = await self.reflexion(
            subtask_id='stage_0.subtask_5',
            reflect_desc=cot_reflect_desc_5,
            n_repeat=2
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results5['answer'])
        logs.append(log5)

    debate_desc_1 = {
        'instruction': (
            "Sub-task 1: Evaluate each candidate product against the refined structural proposal and reaction context. "
            "Input content: results (thinking and answer) from all iterations of stage_0.subtask_5."
        ),
        'final_decision_instruction': (
            "Sub-task 1: Select the best candidate product based on evaluation."
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': [
            'user query',
            'thinking of all stage_0.subtask_5 iterations',
            'answer of all stage_0.subtask_5 iterations'
        ],
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
            "Sub-task 2: Select the candidate that best matches the spectral data and expected reaction outcome as the final product. "
            "Input content: results (thinking and answer) from stage_1.subtask_1."
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
