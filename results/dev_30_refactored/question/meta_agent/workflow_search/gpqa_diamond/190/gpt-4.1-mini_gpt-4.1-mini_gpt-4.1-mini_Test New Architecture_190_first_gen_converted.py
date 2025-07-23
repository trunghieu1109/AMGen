async def forward_190(self, taskInfo):
    logs = []
    loop_results_stage_0 = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                'Sub-task 1: Extract and summarize the given chemical information, including starting material, reagents, and transformations. '
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
        loop_results_stage_0['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results_stage_0['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_agent_desc_2 = {
            'instruction': (
                'Sub-task 2: Analyze the relationships between components and predict the chemical transformations at each step. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 and previous iterations of stage_0.subtask_4'
            ),
            'final_decision_instruction': (
                'Sub-task 2: Synthesize and choose the most consistent answer for analyzing chemical transformations.'
            ),
            'input': [taskInfo] + 
                     loop_results_stage_0['stage_0.subtask_1']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_1']['answer'] + 
                     loop_results_stage_0['stage_0.subtask_4']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_4']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_2, log_2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2,
            n_repeat=self.max_sc
        )
        loop_results_stage_0['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results_stage_0['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Interpret the mechanistic implications of each reaction step, including functional group changes and stereochemical considerations. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_2 and previous iterations of stage_0.subtask_4'
            ),
            'final_decision_instruction': (
                'Sub-task 3: Synthesize and choose the most consistent answer for mechanistic interpretation.'
            ),
            'input': [taskInfo] + 
                     loop_results_stage_0['stage_0.subtask_2']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_2']['answer'] + 
                     loop_results_stage_0['stage_0.subtask_4']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_4']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_3, log_3 = await self.sc_cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3,
            n_repeat=self.max_sc
        )
        loop_results_stage_0['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results_stage_0['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        reflect_desc_4 = {
            'instruction': (
                'Sub-task 4: Refine and consolidate the intermediate conclusions to produce a provisional structure for product 4 with reasoning documentation. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_3 and previous iterations of stage_0.subtask_4'
            ),
            'critic_instruction': (
                'Please review and provide the limitations of provided solutions of the provisional structure for product 4.'
            ),
            'input': [taskInfo] + 
                     loop_results_stage_0['stage_0.subtask_3']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_3']['answer'] + 
                     loop_results_stage_0['stage_0.subtask_4']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_4, log_4 = await self.reflexion(
            subtask_id='stage_0.subtask_4',
            reflect_desc=reflect_desc_4,
            n_repeat=self.max_round
        )
        loop_results_stage_0['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results_stage_0['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        aggregate_desc_5 = {
            'instruction': (
                'Sub-task 5: Aggregate the refined outputs to prepare a comprehensive intermediate result for evaluation. '
                'Input content: all thinking and answers from stage_0.subtask_4'
            ),
            'input': [taskInfo] + 
                     loop_results_stage_0['stage_0.subtask_4']['thinking'] + 
                     loop_results_stage_0['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_5, log_5 = await self.aggregate(
            subtask_id='stage_0.subtask_5',
            aggregate_desc=aggregate_desc_5
        )
        loop_results_stage_0['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results_stage_0['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

    cot_debate_desc_1 = {
        'instruction': (
            'Stage 1 Sub-task 1: Evaluate the candidate product structures against the refined mechanistic conclusions and select the best matching structure. '
            'Input content: taskInfo, all thinking and answers from stage_0.subtask_5'
        ),
        'final_decision_instruction': (
            'Stage 1 Sub-task 1: Synthesize and select the best candidate product structure based on evaluation.'
        ),
        'input': [taskInfo] + 
                 loop_results_stage_0['stage_0.subtask_5']['thinking'] + 
                 loop_results_stage_0['stage_0.subtask_5']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_stage1_1, log_stage1_1 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=cot_debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_1)

    aggregate_desc_2 = {
        'instruction': (
            'Stage 1 Sub-task 2: Aggregate the evaluation results to finalize the best candidate product structure. '
            'Input content: thinking and answer from stage_1.subtask_1'
        ),
        'input': [taskInfo, results_stage1_1['thinking'], results_stage1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_stage1_2, log_stage1_2 = await self.aggregate(
        subtask_id='stage_1.subtask_2',
        aggregate_desc=aggregate_desc_2
    )
    logs.append(log_stage1_2)

    final_answer = await self.make_final_answer(results_stage1_2['thinking'], results_stage1_2['answer'])
    return final_answer, logs
