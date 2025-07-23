async def forward_156(self, taskInfo):
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
                'Sub-task 1: Characterize the retrovirus genome type and confirm it is RNA-based to justify cDNA sequencing. '
                'Input content are results (both thinking and answer) from: none for first iteration, or previous iterations of stage_0.subtask_1.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_agent_desc_2 = {
            'instruction': (
                'Sub-task 2: Perform cDNA sequencing of viral RNA to identify the viral genetic sequence accurately. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_0.subtask_2, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Analyze sequencing data to design primers specific for the retrovirus genome. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_2 & former iterations of stage_0.subtask_3, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_agent_desc_4 = {
            'instruction': (
                'Sub-task 4: Develop a real-time PCR assay protocol using the designed primers for rapid viral detection. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_3 & former iterations of stage_0.subtask_4, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_0_4, log_0_4 = await self.cot(subtask_id='stage_0.subtask_4', cot_agent_desc=cot_agent_desc_4)
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_agent_desc_5 = {
            'instruction': (
                'Sub-task 5: Document the intermediate design steps and prepare data for refinement and evaluation. '
                'Input content are results (both thinking and answer) from: stage_0.subtask_4 & former iterations of stage_0.subtask_5, respectively.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_0_5, log_0_5 = await self.answer_generate(subtask_id='stage_0.subtask_5', cot_agent_desc=cot_agent_desc_5)
        loop_results['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    cot_reflect_desc_1 = {
        'instruction': (
            'Sub-task 1: Review and simplify the intermediate design documentation to enhance clarity and usability. '
            'Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively.'
        ),
        'critic_instruction': (
            'Please review and provide the limitations of the intermediate design documentation and suggest improvements.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.reflexion(subtask_id='stage_1.subtask_1', reflect_desc=cot_reflect_desc_1, n_repeat=self.max_round)
    logs.append(log_1_1)

    debate_desc_1 = {
        'instruction': (
            'Sub-task 2: Evaluate the designed real-time PCR assay against criteria of speed, accuracy, and feasibility. '
            'Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively.'
        ),
        'final_decision_instruction': (
            'Sub-task 2: Select the best evaluation of the real-time PCR assay based on speed, accuracy, and feasibility.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_1_2, log_1_2 = await self.debate(subtask_id='stage_1.subtask_2', debate_desc=debate_desc_1, n_repeat=self.max_round)
    logs.append(log_1_2)

    aggregate_desc_1 = {
        'instruction': (
            'Sub-task 3: Select the best diagnostic approach based on evaluation, prioritizing cDNA sequencing and real-time PCR. '
            'Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_1.subtask_2, respectively.'
        ),
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_1.subtask_1 and stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.aggregate(subtask_id='stage_1.subtask_3', aggregate_desc=aggregate_desc_1)
    logs.append(log_1_3)

    cot_reflect_desc_2 = {
        'instruction': (
            'Sub-task 1: Translate the selected diagnostic design into a detailed molecular kit development protocol. '
            'Input content are results (both thinking and answer) from: stage_1.subtask_3, respectively.'
        ),
        'critic_instruction': (
            'Please review the detailed molecular kit development protocol and suggest improvements or confirm readiness.'
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.reflexion(subtask_id='stage_2.subtask_1', reflect_desc=cot_reflect_desc_2, n_repeat=self.max_round)
    logs.append(log_2_1)

    cot_sc_instruction_2 = (
        'Sub-task 2: Simulate or model the real-time PCR assay performance to predict diagnostic accuracy and speed. '
        'Input content are results (both thinking and answer) from: stage_1.subtask_3, respectively.'
    )
    final_decision_instruction_2 = (
        'Sub-task 2: Synthesize and choose the most consistent simulation results for real-time PCR assay performance.'
    )
    cot_sc_desc_2 = {
        'instruction': cot_sc_instruction_2,
        'final_decision_instruction': final_decision_instruction_2,
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_2, log_2_2 = await self.sc_cot(subtask_id='stage_2.subtask_2', cot_agent_desc=cot_sc_desc_2, n_repeat=self.max_sc)
    logs.append(log_2_2)

    review_desc_1 = {
        'instruction': (
            'Sub-task 1: Assess the developed molecular diagnostic kit protocol for compliance with clinical diagnostic standards. '
            'Input content are results (both thinking and answer) from: stage_2.subtask_1 & stage_2.subtask_2, respectively.'
        ),
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer'], results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_3_1, log_3_1 = await self.review(subtask_id='stage_3.subtask_1', review_desc=review_desc_1)
    logs.append(log_3_1)

    cot_instruction_2 = {
        'instruction': (
            'Sub-task 2: Provide feedback on potential improvements or confirm readiness for practical deployment. '
            'Input content are results (both thinking and answer) from: stage_3.subtask_1, respectively.'
        ),
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_3_2, log_3_2 = await self.cot(subtask_id='stage_3.subtask_2', cot_agent_desc=cot_instruction_2)
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
