async def forward_150(self, taskInfo):
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
                'Normalize the given state vector (-1, 2, 1) to obtain a unit vector representing the system state. '
                'Input content: taskInfo'
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_1, log_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_agent_desc_2 = {
            'instruction': (
                'Calculate the eigenvalues and eigenvectors of the observable matrix P given by ' 
                '[[0, 1/sqrt(2), 0], [1/sqrt(2), 0, 1/sqrt(2)], [0, 1/sqrt(2), 0]]. '
                'Input content: taskInfo'
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_2, log_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_agent_desc_3 = {
            'instruction': (
                'Identify the eigenspace corresponding to the eigenvalue 0 from the eigenvectors computed in stage_0.subtask_2. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_2'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_3, log_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        cot_agent_desc_4 = {
            'instruction': (
                'Project the normalized state vector (from stage_0.subtask_1) onto the eigenspace of eigenvalue 0 (from stage_0.subtask_3) to find the projection vector. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 and stage_0.subtask_3, and previous iterations of stage_0.subtask_4'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
        }
        results_4, log_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        cot_agent_desc_5 = {
            'instruction': (
                'Calculate the squared magnitude of the projection vector (from stage_0.subtask_4) to obtain the probability of measuring eigenvalue 0. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_4, and previous iterations of stage_0.subtask_5'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4', 'thinking of previous stage_0.subtask_5', 'answer of previous stage_0.subtask_5']
        }
        results_5, log_5 = await self.cot(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_5
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

    cot_reflect_desc_1 = {
        'instruction': (
            'Simplify and consolidate the calculated probability expression to a final simplified form. '
            'Input content: all thinking and answers from stage_0.subtask_5'
        ),
        'critic_instruction': (
            'Please review and provide feedback on the simplification and consolidation of the probability expression.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query'] + ['thinking of stage_0.subtask_5'] * len(loop_results['stage_0.subtask_5']['thinking']) + ['answer of stage_0.subtask_5'] * len(loop_results['stage_0.subtask_5']['answer'])
    }
    results_reflex_1, log_reflex_1 = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_reflex_1)

    debate_desc_2 = {
        'instruction': (
            'Evaluate the simplified probability against the given multiple-choice options (1/3, 2/3, sqrt(2/3), 1) to select the best matching answer. '
            'Input content: taskInfo, thinking and answer from stage_1.subtask_1'
        ),
        'final_decision_instruction': (
            'Select the best matching answer from the given choices based on the simplified probability.'
        ),
        'input': [taskInfo, results_reflex_1['thinking'], results_reflex_1['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_debate_2, log_debate_2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_2,
        n_repeat=self.max_round
    )
    logs.append(log_debate_2)

    cot_reflect_desc_3 = {
        'instruction': (
            'Apply any necessary mathematical transformations or checks to confirm the correctness of the selected probability value. '
            'Input content: taskInfo, thinking and answer from stage_1.subtask_2'
        ),
        'critic_instruction': (
            'Please review and confirm the correctness of the selected probability value and transformations applied.'
        ),
        'input': [taskInfo, results_debate_2['thinking'], results_debate_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_reflex_3, log_reflex_3 = await self.reflexion(
        subtask_id='stage_2.subtask_1',
        reflect_desc=cot_reflect_desc_3,
        n_repeat=self.max_round
    )
    logs.append(log_reflex_3)

    review_desc_4 = {
        'instruction': (
            'Review the final probability calculation and selected answer for consistency, correctness, and alignment with quantum measurement theory. '
            'Input content: taskInfo, thinking and answer from stage_2.subtask_1'
        ),
        'input': [taskInfo, results_reflex_3['thinking'], results_reflex_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_review_4, log_review_4 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_4
    )
    logs.append(log_review_4)

    final_answer = await self.make_final_answer(results_review_4['thinking'], results_review_4['answer'])
    return final_answer, logs
