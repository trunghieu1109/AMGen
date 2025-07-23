async def forward_153(self, taskInfo):
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
                'Sub-task 1: Extract and summarize all given spectral and molecular data from the query, including mass spectrometry, IR, and 1H NMR information. '
                'Input content: taskInfo'
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
                'Sub-task 2: Analyze relationships between the extracted spectral components to infer functional groups, isotopic patterns, and substitution patterns. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_0.subtask_5, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
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
                'Sub-task 3: Identify the chemical field and subfields relevant to the problem to contextualize the spectral data interpretation. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_2 & former iterations of stage_0.subtask_5, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
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
                'Sub-task 4: Highlight and document any ambiguities or missing information in the spectral data that could affect structural identification. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_3 & former iterations of stage_0.subtask_5, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results4, log4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

        cot_reflect_desc_5 = {
            'instruction': (
                'Sub-task 5: Integrate all previous analyses to generate a provisional structural suggestion and reasoning for the unidentified compound. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_4 & stage_0.subtask_2 & former iterations of stage_0.subtask_5, respectively.'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of Sub-task 5 to improve the structural suggestion.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4',
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'
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
            'Sub-task 1: Evaluate the provided candidate compounds against the refined structural suggestion and spectral data to select the best matching candidate. '
            'Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Select the best matching candidate compound based on spectral data and structural suggestion.'
        ),
        'input': [taskInfo, loop_results['stage_0.subtask_5']['thinking'][-1], loop_results['stage_0.subtask_5']['answer'][-1]],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=2
    )
    logs.append(log_stage1)

    final_answer = await self.make_final_answer(results_stage1['thinking'], results_stage1['answer'])
    return final_answer, logs
