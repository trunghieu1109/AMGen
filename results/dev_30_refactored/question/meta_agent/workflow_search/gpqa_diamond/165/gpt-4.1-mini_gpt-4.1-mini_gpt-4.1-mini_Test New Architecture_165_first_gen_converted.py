async def forward_165(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                'Sub-task 1: Extract and summarize all given physical and mathematical information from the Lagrangian and model setup relevant to the pseudo-Goldstone boson mass calculation. '
                'Input: taskInfo containing the query with Lagrangian, fields, VEVs, and candidate formulae.'
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
                'Sub-task 2: Analyze the relationships between fields, VEVs, and mass terms to identify contributions to the pseudo-Goldstone boson mass from radiative corrections. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_1 and stage_0.subtask_3 from all previous iterations.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
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
                'Sub-task 3: Compile and compare the candidate mass formulae, noting differences in terms, coefficients, and denominators to prepare for selection. '
                'Input: taskInfo, all previous thinking and answers from stage_0.subtask_2 and stage_0.subtask_3 from all previous iterations.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results3, log3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

    cot_reflect_desc_1 = {
        'instruction': (
            'Sub-task 1: Simplify and consolidate the intermediate findings to clarify the physical meaning and mathematical correctness of each candidate formula. '
            'Input: all thinking and answers from stage_0.subtask_3 from all iterations.'
        ),
        'critic_instruction': (
            'Please review and provide the limitations of the provided solutions regarding the simplification and consolidation of candidate formulae.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
    }
    results_reflexion, log_reflexion = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1,
        n_repeat=2
    )
    logs.append(log_reflexion)

    debate_desc_2 = {
        'instruction': (
            'Sub-task 2: Evaluate each candidate formula against theoretical consistency criteria such as correct dependence on VEVs and inclusion of relevant particle contributions. '
            'Input: taskInfo, thinking and answer from stage_1.subtask_1.'
        ),
        'final_decision_instruction': 'Sub-task 2: Evaluate and select the most theoretically consistent candidate formula.',
        'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_debate, log_debate = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_2,
        n_repeat=2
    )
    logs.append(log_debate)

    aggregate_desc_3 = {
        'instruction': (
            'Sub-task 3: Select the best candidate formula for the pseudo-Goldstone boson mass based on the evaluation. '
            'Input: taskInfo, thinking and answer from stage_1.subtask_2.'
        ),
        'input': [taskInfo, results_debate['thinking'], results_debate['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_aggregate, log_aggregate = await self.aggregate(
        subtask_id='stage_1.subtask_3',
        aggregate_desc=aggregate_desc_3
    )
    logs.append(log_aggregate)

    cot_reflect_desc_2 = {
        'instruction': (
            'Sub-task 1: Apply any necessary algebraic or conceptual transformations to the selected formula to express it in a final, clear, and standard form. '
            'Input: taskInfo, thinking and answer from stage_1.subtask_3.'
        ),
        'critic_instruction': (
            'Please review and provide feedback on the clarity and correctness of the transformed final formula.'
        ),
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_transform, log_transform = await self.reflexion(
        subtask_id='stage_2.subtask_1',
        reflect_desc=cot_reflect_desc_2,
        n_repeat=1
    )
    logs.append(log_transform)

    review_desc_1 = {
        'instruction': (
            'Sub-task 1: Validate the final mass formula for the pseudo-Goldstone boson by checking consistency with known theoretical expectations and physical plausibility. '
            'Input: taskInfo, thinking and answer from stage_1.subtask_3 and stage_2.subtask_1.'
        ),
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer'], results_transform['thinking'], results_transform['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_review, log_review = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log_review)

    final_answer = await self.make_final_answer(
        results_review['thinking'],
        results_review['answer']
    )

    return final_answer, logs
