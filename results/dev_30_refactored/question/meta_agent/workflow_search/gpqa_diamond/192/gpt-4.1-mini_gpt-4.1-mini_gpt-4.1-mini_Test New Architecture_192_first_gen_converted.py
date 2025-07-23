async def forward_192(self, taskInfo):
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
                'Sub-task 1: Extract and restate the given relationship between number of stars and parallax, '
                'and express parallax in terms of distance r. Input: user query from taskInfo.'
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
                'Sub-task 2: Rewrite the star count function from parallax variable to distance variable by substituting plx = 1/r. '
                'Input: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_0.subtask_2.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_2', 'answer of previous stage_0.subtask_2']
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
                'Sub-task 3: Determine the differential relationship between parallax and distance (compute d(plx)/d(r)) to understand how the distribution transforms. '
                'Input: results (thinking and answer) from stage_0.subtask_2 and all previous iterations of stage_0.subtask_3.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_3']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_3', 'answer of previous stage_0.subtask_3']
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
                'Sub-task 4: Apply the change of variables formula to convert the star count per unit parallax to star count per unit distance, incorporating the Jacobian of the transformation. '
                'Input: results (thinking and answer) from stage_0.subtask_3 and all previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_4']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
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
                'Sub-task 5: Summarize the intermediate expression for the number of stars per unit distance as a function of r, preparing for simplification. '
                'Input: results (thinking and answer) from stage_0.subtask_4 and all previous iterations of stage_0.subtask_5.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_5']['answer'] + loop_results['stage_0.subtask_5']['thinking'],
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
            'Sub-task 1: Simplify the intermediate expression obtained for star count per unit distance to a power-law form in r. '
            'Input: results (thinking and answer) from stage_0.subtask_5.'
        ),
        'critic_instruction': (
            'Please review and provide the limitations of the simplification and suggest improvements if any.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_reflexion_1, log_reflexion_1 = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1,
        n_repeat=2
    )
    logs.append(log_reflexion_1)

    debate_desc_2 = {
        'instruction': (
            'Sub-task 2: Compare the simplified power-law form with the given answer choices to identify the best matching option. '
            'Input: results (thinking and answer) from stage_1.subtask_1.'
        ),
        'final_decision_instruction': (
            'Sub-task 2: Synthesize and select the best matching answer choice for the star count power-law.'
        ),
        'input': [taskInfo, results_reflexion_1['thinking'], results_reflexion_1['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_debate_2, log_debate_2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_2,
        n_repeat=2
    )
    logs.append(log_debate_2)

    aggregate_desc_3 = {
        'instruction': (
            'Sub-task 3: Consolidate reasoning and select the final candidate answer based on the comparison. '
            'Input: results (thinking and answer) from stage_1.subtask_2.'
        ),
        'input': [taskInfo, results_debate_2['thinking'], results_debate_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_aggregate_3, log_aggregate_3 = await self.aggregate(
        subtask_id='stage_1.subtask_3',
        aggregate_desc=aggregate_desc_3
    )
    logs.append(log_aggregate_3)

    cot_sc_desc_4 = {
        'instruction': (
            'Sub-task 1: Explicitly state the final formula for the number of stars per unit distance r, confirming the power-law exponent. '
            'Input: results (thinking and answer) from stage_1.subtask_3.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Provide a consistent and explicit final formula for star count per unit distance r.'
        ),
        'input': [taskInfo, results_aggregate_3['thinking'], results_aggregate_3['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_sc_4, log_sc_4 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_sc_desc_4,
        n_repeat=3
    )
    logs.append(log_sc_4)

    review_desc_5 = {
        'instruction': (
            'Sub-task 1: Evaluate the correctness and consistency of the final answer with astrophysical principles and problem constraints. '
            'Input: results (thinking and answer) from stage_2.subtask_1.'
        ),
        'input': [taskInfo, results_sc_4['thinking'], results_sc_4['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_review_5, log_review_5 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_5
    )
    logs.append(log_review_5)

    final_answer = await self.make_final_answer(results_sc_4['thinking'], results_sc_4['answer'])
    return final_answer, logs
