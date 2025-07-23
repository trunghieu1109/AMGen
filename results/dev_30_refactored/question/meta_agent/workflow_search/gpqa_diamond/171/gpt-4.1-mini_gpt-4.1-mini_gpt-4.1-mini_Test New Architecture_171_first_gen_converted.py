async def forward_171(self, taskInfo):
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
                'Sub-task 1: Extract and summarize all given information from the query relevant to the excitation ratio and energy difference. '
                'Input content: taskInfo'
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_agent_desc_2 = {
            'instruction': (
                'Sub-task 2: Analyze the physical relationship between excitation ratio, energy difference, and temperatures using the Boltzmann distribution under LTE assumptions. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 and previous iterations of stage_0.subtask_2'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_2', 'answer of previous stage_0.subtask_2']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Derive the mathematical expression relating ln(2) to T_1, T_2, and ΔE/k based on the Boltzmann formula. '
                'Input content: all thinking and answers from stage_0.subtask_2 and previous iterations of stage_0.subtask_3'
            ),
            'input': loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_3', 'answer of previous stage_0.subtask_3']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_agent_desc_4 = {
            'instruction': (
                'Sub-task 4: Compare the derived expression with the candidate equations to identify structural similarities and differences. '
                'Input content: all thinking and answers from stage_0.subtask_3 and previous iterations of stage_0.subtask_4'
            ),
            'input': loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_agent_desc_5 = {
            'instruction': (
                'Sub-task 5: Document reasoning and preliminary conclusions about which candidate equations could be correct based on the derivation. '
                'Input content: all thinking and answers from stage_0.subtask_4 and previous iterations of stage_0.subtask_5'
            ),
            'input': loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4', 'thinking of previous stage_0.subtask_5', 'answer of previous stage_0.subtask_5']
        }
        results_0_5, log_0_5 = await self.answer_generate(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_5
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    cot_reflect_desc_1 = {
        'instruction': (
            'Sub-task 1: Simplify and consolidate the preliminary expressions and reasoning from stage_0 to clarify the correct form of the temperature relation. '
            'Input content: all thinking and answers from stage_0.subtask_5'
        ),
        'critic_instruction': (
            'Please review and provide the limitations of the simplification and consolidation of the temperature relation expressions.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.reflexion(
        subtask_id='stage_1.subtask_1',
        reflect_desc=cot_reflect_desc_1,
        n_repeat=self.max_round if hasattr(self, 'max_round') else 2
    )
    logs.append(log_1_1)

    debate_desc_2 = {
        'instruction': (
            'Sub-task 2: Evaluate each candidate equation against the derived expression and physical constraints to select the best matching equation. '
            'Input content: thinking and answer from stage_1.subtask_1'
        ),
        'final_decision_instruction': (
            'Sub-task 2: Synthesize and select the best candidate equation matching the physical and mathematical derivation.'
        ),
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id='stage_1.subtask_2',
        debate_desc=debate_desc_2,
        n_repeat=self.max_round if hasattr(self, 'max_round') else 2
    )
    logs.append(log_1_2)

    aggregate_desc_3 = {
        'instruction': (
            'Sub-task 3: Aggregate the evaluation results to finalize the selection of the correct candidate equation. '
            'Input content: thinking and answer from stage_1.subtask_2'
        ),
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.aggregate(
        subtask_id='stage_1.subtask_3',
        aggregate_desc=aggregate_desc_3
    )
    logs.append(log_1_3)

    cot_reflect_desc_2 = {
        'instruction': (
            'Sub-task 1: Apply algebraic transformations to the selected candidate equation to express it in a clear and standard form. '
            'Input content: thinking and answer from stage_1.subtask_3'
        ),
        'critic_instruction': (
            'Please review the algebraic transformations for correctness and clarity.'
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id='stage_2.subtask_1',
        reflect_desc=cot_reflect_desc_2,
        n_repeat=self.max_round if hasattr(self, 'max_round') else 2
    )
    logs.append(log_2_1)

    cot_sc_desc_2 = {
        'instruction': (
            'Sub-task 2: Derive any additional implications or simplified forms useful for interpretation or further use. '
            'Input content: thinking and answer from stage_2.subtask_1'
        ),
        'final_decision_instruction': (
            'Sub-task 2: Synthesize the most consistent simplified forms and implications.'
        ),
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.sc_cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_sc_desc_2,
        n_repeat=self.max_sc if hasattr(self, 'max_sc') else 3
    )
    logs.append(log_2_2)

    review_desc_1 = {
        'instruction': (
            'Sub-task 1: Evaluate the final transformed equation for correctness, consistency with physical principles, and alignment with the problem statement. '
            'Input content: thinking and answer from stage_1.subtask_3 and stage_2.subtask_2'
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer'], results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log_3_1)

    cot_agent_desc_2 = {
        'instruction': (
            'Sub-task 2: Produce a final assessment and confirm the correct candidate equation for the effective temperatures T_1 and T_2. '
            'Input content: thinking and answer from stage_3.subtask_1'
        ),
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_3_2, log_3_2 = await self.cot(
        subtask_id='stage_3.subtask_2',
        cot_agent_desc=cot_agent_desc_2
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])

    return final_answer, logs
