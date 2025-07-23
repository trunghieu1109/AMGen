async def forward_159(self, taskInfo):
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
                'Sub-task 1: Extract and summarize all given information and assumptions from the query, including the aperture shape, wavelength, and diffraction context. '
                'Input content: the original query dictionary containing question and choices.'
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
                'Sub-task 2: Analyze the physical and mathematical relationships governing the diffraction pattern, focusing on the limit as N approaches infinity and applying the small angle approximation. '
                'Input content: results (thinking and answer) from stage_0.subtask_1 from all previous iterations and the current iteration.'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of Sub-task 2, considering the physical and mathematical correctness.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results2, log2 = await self.reflexion(
            subtask_id='stage_0.subtask_2',
            reflect_desc=cot_reflect_desc_2,
            n_repeat=1
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Derive the expression for the angular positions of the first two minima in the diffraction pattern for a circular aperture of radius a. '
                'Input content: results (thinking and answer) from stage_0.subtask_2 from all previous iterations and the current iteration.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
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
                'Sub-task 4: Simplify and refine the derived expression to match the form of the given answer choices, ensuring clarity and correctness. '
                'Input content: results (thinking and answer) from stage_0.subtask_3 and all previous iterations of stage_0.subtask_4.'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of Sub-task 4, focusing on simplification and matching answer choices.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
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
            'Sub-task 1: Evaluate the refined angular distance expression against the provided answer choices and select the best matching candidate. '
            'Input content: results (thinking and answer) from all iterations of stage_0.subtask_4.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Synthesize and select the best matching answer choice for the angular distance between the first two minima.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'],
        'temperature': 0.5
    }
    results_debate, log_debate = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=2
    )
    logs.append(log_debate)

    review_desc_1 = {
        'instruction': (
            'Sub-task 1: Validate the selected answer for correctness, consistency with physical principles, and compliance with the problem assumptions. '
            'Input content: results (thinking and answer) from stage_1.subtask_1.'
        ),
        'input': [taskInfo, results_debate['thinking'], results_debate['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_review, log_review = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log_review)

    final_answer = await self.make_final_answer(results_review['thinking'], results_debate['answer'])
    return final_answer, logs
