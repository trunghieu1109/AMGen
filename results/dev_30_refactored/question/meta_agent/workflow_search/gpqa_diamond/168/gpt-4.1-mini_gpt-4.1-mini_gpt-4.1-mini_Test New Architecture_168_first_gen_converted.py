async def forward_168(self, taskInfo):
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
                'Sub-task 1: Extract and summarize the given information about the original and variant decay processes, '
                'focusing on particle types, energy spectra, and endpoint values. Input content: user query.'
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
                'Sub-task 2: Analyze the kinematic and energy-momentum conservation implications of replacing two V particles with one massless M particle on the energy spectrum of E particles. '
                'Input content: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
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
                'Sub-task 3: Evaluate how the continuous nature of the original E particle spectrum might be affected by the change in emitted particles, considering phase space and endpoint energy Q. '
                'Input content: results (thinking and answer) from stage_0.subtask_2 and all previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
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
                'Sub-task 4: Refine the intermediate analysis by integrating insights from previous subtasks and prior iteration outputs to produce a clearer, more concise understanding of the spectrum changes. '
                'Input content: results (thinking and answer) from stage_0.subtask_1, stage_0.subtask_2, stage_0.subtask_3, and all previous iterations of stage_0.subtask_4.'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of spectrum changes analysis, focusing on clarity, completeness, and physical consistency.'
            ),
            'input': [taskInfo] + 
                     loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + 
                     loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + 
                     loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + 
                     loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1',
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3',
                'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'
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
            'Sub-task 1: Evaluate the candidate answers (choices 1 to 4) against the refined analysis from stage_0 to identify which best matches the expected changes in the E particle energy spectrum. '
            'Input content: results (thinking and answer) from stage_0.subtask_4.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Select the best candidate answer that matches the refined analysis of the E particle energy spectrum changes.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'],
        'temperature': 0.5
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=1
    )
    logs.append(log_stage1)

    review_desc_1 = {
        'instruction': (
            'Sub-task 1: Validate the selected candidate answer for consistency with physical principles of nuclear decay, energy conservation, and known behavior of massless particle emissions. '
            'Input content: results (thinking and answer) from stage_1.subtask_1.'
        ),
        'input': [taskInfo, results_stage1['thinking'], results_stage1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_stage2, log_stage2 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage1['thinking'], results_stage1['answer'])
    return final_answer, logs
