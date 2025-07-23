async def forward_164(self, taskInfo):
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
                'Sub-task 1: Extract and summarize all relevant information from the query, including catalyst types, polymerization conditions, and statements to be evaluated. '
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
                'Sub-task 2: Analyze the relationships and chemical feasibility between catalyst systems, activators, and polymer branching mechanisms based on the extracted information. '
                'Input content: results from stage_0.subtask_1 and all previous iterations of stage_0.subtask_5 thinking and answer'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_5', 'answer of previous stage_0.subtask_5']
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
                'Sub-task 3: Identify and clarify ambiguous or missing details in the statements, such as the nature of the essential additional reaction step and catalyst compatibility. '
                'Input content: results from stage_0.subtask_2 and all previous iterations of stage_0.subtask_4 thinking and answer'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
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
                'Sub-task 4: Integrate industrial and economic considerations, such as catalyst cost and scalability, into the analysis of the statements. '
                'Input content: results from stage_0.subtask_3 and all previous iterations of stage_0.subtask_2 thinking and answer'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_2', 'answer of previous stage_0.subtask_2']
        }
        results_4, log_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        cot_reflect_desc_5 = {
            'instruction': (
                'Sub-task 5: Refine the intermediate reasoning by synthesizing all previous subtasks outputs to form a coherent assessment framework for evaluating the four statements. '
                'Input content: results from stage_0.subtask_4 and all previous iterations of stage_0.subtask_3 thinking and answer'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of stage_0.subtask_5 to improve the assessment framework.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4', 'thinking of previous stage_0.subtask_3', 'answer of previous stage_0.subtask_3']
        }
        results_5, log_5 = await self.reflexion(
            subtask_id='stage_0.subtask_5',
            reflect_desc=cot_reflect_desc_5,
            n_repeat=2
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

    debate_desc_1 = {
        'instruction': (
            'Sub-task 1: Evaluate each of the four statements against the refined assessment framework developed in stage_0 to determine their correctness. '
            'Input content: results from all iterations of stage_0.subtask_5 thinking and answer'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Synthesize the debate and choose the most correct statement regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_1_stage1, log_1_stage1 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=2
    )
    logs.append(log_1_stage1)

    aggregate_desc_2 = {
        'instruction': (
            'Sub-task 2: Select the statement that best aligns with the chemical, industrial, and economic criteria established, justifying the choice. '
            'Input content: results from stage_1.subtask_1 thinking and answer'
        ),
        'input': [taskInfo, results_1_stage1['thinking'], results_1_stage1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_stage1, log_2_stage1 = await self.aggregate(
        subtask_id='stage_1.subtask_2',
        aggregate_desc=aggregate_desc_2
    )
    logs.append(log_2_stage1)

    review_desc_1_stage2 = {
        'instruction': (
            'Sub-task 1: Critically review the selected statement for consistency, correctness, and alignment with known polymer chemistry principles and industrial practices. '
            'Input content: results from stage_1.subtask_2 thinking and answer'
        ),
        'input': [taskInfo, results_2_stage1['thinking'], results_2_stage1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_1_stage2, log_1_stage2 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_1_stage2
    )
    logs.append(log_1_stage2)

    cot_agent_desc_2_stage2 = {
        'instruction': (
            'Sub-task 2: Provide a final assessment confirming or rejecting the selected statement as the correct answer to the query. '
            'Input content: results from stage_2.subtask_1 thinking and answer'
        ),
        'input': [taskInfo, results_1_stage2['thinking'], results_1_stage2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_stage2, log_2_stage2 = await self.cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_agent_desc_2_stage2
    )
    logs.append(log_2_stage2)

    final_answer = await self.make_final_answer(results_2_stage2['thinking'], results_2_stage2['answer'])

    return final_answer, logs
