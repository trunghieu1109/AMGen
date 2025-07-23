async def forward_193(self, taskInfo):
    logs = []
    loop_results_stage_0 = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_0_1 = {
            'instruction': (
                'Sub-task 1: Enumerate all possible spin configurations (S1, S2, S3) where each spin is ±1. '
                'Input content: taskInfo only.'
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results_stage_0['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results_stage_0['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_agent_desc_0_2 = {
            'instruction': (
                'Sub-task 2: Calculate the energy E for each spin configuration using E = -J (S1 S2 + S1 S3 + S2 S3). '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 and previous iterations of stage_0.subtask_2.'
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_1']['answer'] + loop_results_stage_0['stage_0.subtask_1']['thinking'] + loop_results_stage_0['stage_0.subtask_2']['answer'] + loop_results_stage_0['stage_0.subtask_2']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'previous thinking of stage_0.subtask_2', 'previous answer of stage_0.subtask_2']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results_stage_0['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results_stage_0['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_agent_desc_0_3 = {
            'instruction': (
                'Sub-task 3: Group spin configurations by their energy values and count degeneracies for each energy level. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_2 and previous iterations of stage_0.subtask_3.'
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_2']['answer'] + loop_results_stage_0['stage_0.subtask_2']['thinking'] + loop_results_stage_0['stage_0.subtask_3']['answer'] + loop_results_stage_0['stage_0.subtask_3']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'previous thinking of stage_0.subtask_3', 'previous answer of stage_0.subtask_3']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_0_3
        )
        loop_results_stage_0['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results_stage_0['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_agent_desc_0_4 = {
            'instruction': (
                'Sub-task 4: Express the partition function Z as a sum over energy groups: Z = Σ g(E) e^{-β E}, where g(E) is degeneracy. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_3 and previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_3']['answer'] + loop_results_stage_0['stage_0.subtask_3']['thinking'] + loop_results_stage_0['stage_0.subtask_4']['answer'] + loop_results_stage_0['stage_0.subtask_4']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'previous thinking of stage_0.subtask_4', 'previous answer of stage_0.subtask_4']
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_0_4
        )
        loop_results_stage_0['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results_stage_0['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

    cot_reflect_desc_1_1 = {
        'instruction': (
            'Sub-task 1: Simplify and consolidate the partition function expression obtained from stage_0.subtask_4 into a closed-form expression. '
            'Input content: taskInfo and all thinking and answers from stage_0.subtask_4.'
        ),
        'critic_instruction': (
            'Please review and provide feedback on the simplification and consolidation of the partition function expression.'
        ),
        'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_4']['thinking'] + loop_results_stage_0['stage_0.subtask_4']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
    }
    results_1_1, log_1_1 = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1_1,
        n_repeat=2
    )
    logs.append(log_1_1)

    debate_desc_1_2 = {
        'instruction': (
            'Sub-task 2: Compare the simplified partition function expression against the provided candidate choices to identify matching forms. '
            'Input content: taskInfo and thinking and answer from stage_1.subtask_1.'
        ),
        'final_decision_instruction': (
            'Sub-task 2: Select the candidate choice that best matches the derived partition function based on correctness and form.'
        ),
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_1_2,
        n_repeat=2
    )
    logs.append(log_1_2)

    aggregate_desc_1_3 = {
        'instruction': (
            'Sub-task 3: Select the best candidate expression for Z that matches the derived partition function based on correctness and form. '
            'Input content: taskInfo and thinking and answer from stage_1.subtask_2.'
        ),
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.aggregate(
        subtask_id='stage_1.subtask_3',
        aggregate_desc=aggregate_desc_1_3
    )
    logs.append(log_1_3)

    cot_reflect_desc_2_1 = {
        'instruction': (
            'Sub-task 1: Apply any necessary algebraic transformations or factorization to the selected partition function expression for clarity. '
            'Input content: taskInfo and thinking and answer from stage_1.subtask_3.'
        ),
        'critic_instruction': (
            'Please review and provide feedback on the algebraic transformations applied for clarity.'
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id='stage_2.subtask_1',
        reflect_desc=cot_reflect_desc_2_1,
        n_repeat=2
    )
    logs.append(log_2_1)

    review_desc_3_1 = {
        'instruction': (
            'Sub-task 1: Evaluate the final partition function expression for physical consistency and correctness against known properties. '
            'Input content: taskInfo, thinking and answer from stage_1.subtask_3 and stage_2.subtask_1.'
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer'], results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_1_3['thinking'], results_1_3['answer'])
    return final_answer, logs
