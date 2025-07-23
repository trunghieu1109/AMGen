async def forward_157(self, taskInfo):
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
                'Sub-task 1: Extract and summarize all relevant information from the query, including mutation locations, mutation types, protein domains, and functional consequences. '
                'Input content: taskInfo only.'
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results1, log1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_1)
        logs.append(log1)
        loop_results['stage_0.subtask_1']['thinking'].append(results1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results1['answer'])

        cot_agent_desc_2 = {
            'instruction': (
                'Sub-task 2: Analyze the relationships between phosphorylation, dimerization, nuclear translocation, and transcriptional activation, integrating the summarized information. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 and stage_0.subtask_5 from all prior iterations.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results2, log2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_2)
        logs.append(log2)
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Map the functional roles of the transactivation and dimerization domains and interpret how mutations X and Y affect these functions based on their domain locations and mutation types. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_2 and stage_0.subtask_5 from all prior iterations.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results3, log3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_3)
        logs.append(log3)
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])

        cot_agent_desc_4 = {
            'instruction': (
                'Sub-task 4: Interpret the molecular mechanism of the dominant-negative mutation Y in the dimerization domain, considering typical dominant-negative effects on protein complexes. '
                'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_3 and stage_0.subtask_5 from all prior iterations.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results4, log4 = await self.cot(subtask_id='stage_0.subtask_4', cot_agent_desc=cot_agent_desc_4)
        logs.append(log4)
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])

        debate_instruction_5 = (
            'Sub-task 5: Evaluate each provided molecular phenotype option in the context of the dominant-negative mutation Y and select the most plausible molecular phenotype based on the integrated analysis. '
            'Input content: taskInfo, all previous thinking and answers from stage_0.subtask_4 from all prior iterations.'
        )
        final_decision_instruction_5 = (
            'Sub-task 5: Select the most plausible molecular phenotype option for mutation Y based on the integrated analysis from stage_0.subtask_4.'
        )
        debate_desc_5 = {
            'instruction': debate_instruction_5,
            'final_decision_instruction': final_decision_instruction_5,
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'],
            'temperature': 0.5
        }
        results5, log5 = await self.debate(subtask_id='stage_0.subtask_5', debate_desc=debate_desc_5, n_repeat=1)
        logs.append(log5)
        loop_results['stage_0.subtask_5']['thinking'].append(results5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results5['answer'])

    debate_instruction_stage1 = (
        'Stage 1 Sub-task 1: Select the best molecular phenotype candidate from the evaluated options, based on the refined outputs from stage_0.subtask_5.'
    )
    final_decision_instruction_stage1 = (
        'Stage 1 Sub-task 1: Choose the best molecular phenotype option for mutation Y based on all integrated analyses from stage_0.subtask_5.'
    )
    debate_desc_stage1 = {
        'instruction': debate_instruction_stage1,
        'final_decision_instruction': final_decision_instruction_stage1,
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_stage1, log_stage1 = await self.debate(subtask_id='stage_1.subtask_1', debate_desc=debate_desc_stage1, n_repeat=1)
    logs.append(log_stage1)

    final_answer = await self.make_final_answer(results_stage1['thinking'], results_stage1['answer'])
    return final_answer, logs
