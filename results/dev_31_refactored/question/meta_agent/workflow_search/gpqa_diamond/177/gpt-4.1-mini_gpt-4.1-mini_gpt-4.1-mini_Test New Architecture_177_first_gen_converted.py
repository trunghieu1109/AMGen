async def forward_177(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                'Sub-task 1: Determine the canonical mass dimensions of the fields psi and F^{mu nu} and confirm the dimensionlessness of sigma_{mu nu}. '
                'Input content: the query question and context.'
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
                'Sub-task 2: Calculate the mass dimension of the coupling constant kappa using the requirement that the interaction Lagrangian has mass dimension 4. '
                'Input content: the query question, thinking and answer from stage_0.subtask_1.'
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
                'Sub-task 3: Analyze the renormalizability of the theory based on the mass dimension of kappa and standard QFT criteria. '
                'Input content: the query question, thinking and answer from stage_0.subtask_2.'
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

    cot_sc_desc_1 = {
        'instruction': (
            'Sub-task 1: Evaluate the four given multiple-choice options against the derived mass dimension and renormalizability conclusions to select the best matching candidate. '
            'Input content: the query question, and all thinking and answers from stage_0.subtask_3 across all iterations.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Synthesize and choose the most consistent answer for the best matching candidate based on previous analysis.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
        'temperature': 0.5,
        'context_desc': ['user query'] + ['thinking of stage_0.subtask_3']*len(loop_results['stage_0.subtask_3']['thinking']) + ['answer of stage_0.subtask_3']*len(loop_results['stage_0.subtask_3']['answer'])
    }
    results4, log4 = await self.sc_cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_sc_desc_1,
        n_repeat=self.max_sc if hasattr(self, 'max_sc') else 3
    )
    logs.append(log4)

    cot_reflect_desc_1 = {
        'instruction': (
            'Sub-task 1: Validate the selected candidate for correctness, consistency, and alignment with QFT principles, providing final assessment. '
            'Input content: the query question, thinking and answer from stage_0.subtask_3 and stage_1.subtask_1.'
        ),
        'critic_instruction': (
            'Please review and provide the limitations of the selected candidate solution and its consistency with QFT principles.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + [results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context_desc': ['user query'] + ['thinking of stage_0.subtask_3']*len(loop_results['stage_0.subtask_3']['thinking']) + ['answer of stage_0.subtask_3']*len(loop_results['stage_0.subtask_3']['answer']) + ['thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results5, log5 = await self.reflexion(
        subtask_id='stage_2.subtask_1',
        reflect_desc=cot_reflect_desc_1,
        n_repeat=getattr(self, 'max_round', 1)
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
