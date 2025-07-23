async def forward_160(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                'Sub-task 1: Extract and summarize all relevant given information from the query, including physical parameters and observed phenomena. '
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
                'Sub-task 2: Analyze the physical meaning of the initial mean free path λ1 under ultra-high vacuum conditions without electron beam influence. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_1 and previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
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
                'Sub-task 3: Analyze the effect of electron beam initiation on gas molecule scattering and how it modifies the effective mean free path to λ2. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_2 and previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
        }
        results3, log3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

        cot_reflect_desc_4 = {
            'instruction': (
                'Sub-task 4: Integrate the analyses to hypothesize the quantitative relationship between λ2 and λ1, considering electron scattering effects and constant temperature. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_3, stage_0.subtask_2, stage_0.subtask_1, and previous iterations of stage_0.subtask_4.'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of the relationship between λ2 and λ1 considering electron scattering and constant temperature.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3',
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1',
                'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4'
            ]
        }
        results4, log4 = await self.reflexion(
            subtask_id='stage_0.subtask_4',
            reflect_desc=cot_reflect_desc_4,
            n_repeat=1
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

    debate_desc_1 = {
        'instruction': (
            'Sub-task 1: Evaluate the candidate answer choices against the refined hypothesis about λ2 and select the best fitting conclusion. '
            'Input: taskInfo, final thinking and answer from stage_0.subtask_4.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Select the best fitting conclusion about λ2 based on the refined hypothesis and analyses.'
        ),
        'input': [taskInfo, loop_results['stage_0.subtask_4']['thinking'][-1], loop_results['stage_0.subtask_4']['answer'][-1]],
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=1
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
