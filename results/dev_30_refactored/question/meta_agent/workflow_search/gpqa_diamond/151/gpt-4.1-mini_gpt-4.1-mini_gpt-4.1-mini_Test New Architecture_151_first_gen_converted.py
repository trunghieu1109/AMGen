async def forward_151(self, taskInfo):
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
                'Sub-task 1: Extract and summarize the key biological information from the query, including the peptide treatment, '
                'shmoo formation, and chromatin proteome analysis context. Input: taskInfo containing the question and choices.'
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
                'Sub-task 2: Analyze the roles and relevance of each protein complex option (pre-initiation, pre-replication, enhancer, nucleosome histone) '
                'in the context of active chromatin during shmoo formation. Input: taskInfo, all previous thinking and answers from stage_0.subtask_1 and previous iterations of stage_0.subtask_2.'
            ),
            'final_decision_instruction': (
                'Sub-task 2: Synthesize and choose the most consistent answer for the analysis of protein complexes in active chromatin during shmoo formation.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_2', 'answer of previous stage_0.subtask_2']
        }
        results2, log2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_agent_desc_3 = {
            'instruction': (
                'Sub-task 3: Evaluate which protein complex is least likely to be present in the active chromatin proteome under the experimental conditions described. '
                'Input: taskInfo, all thinking and answers from stage_0.subtask_2 and previous iterations of stage_0.subtask_3.'
            ),
            'final_decision_instruction': (
                'Sub-task 3: Synthesize and choose the most consistent answer for the least represented protein complex in active chromatin proteome.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_0.subtask_3', 'answer of previous stage_0.subtask_3']
        }
        results3, log3 = await self.sc_cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

        cot_agent_desc_4 = {
            'instruction': (
                'Sub-task 4: Document the reasoning process and preliminary conclusion about the least represented protein complex in the assay. '
                'Input: taskInfo, all thinking and answers from stage_0.subtask_3 and previous iterations of stage_0.subtask_4.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4']
        }
        results4, log4 = await self.answer_generate(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

    cot_agent_desc_5 = {
        'instruction': (
            'Sub-task 1: Combine the summarized biological context and analysis from stage_0 to form a consolidated understanding of the query. '
            'Input: all thinking and answers from stage_0.subtask_4.'
        ),
        'input': loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
    }
    results5, log5 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=cot_agent_desc_5
    )
    logs.append(log5)

    cot_agent_desc_6 = {
        'instruction': (
            'Sub-task 2: Integrate knowledge about chromatin immunoprecipitation and mass spectrometry relevance to the protein complexes involved. '
            'Input: all thinking and answers from stage_0.subtask_4.'
        ),
        'input': loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
    }
    results6, log6 = await self.cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_6
    )
    logs.append(log6)

    cot_sc_instruction_7 = (
        'Sub-task 1: Identify and select the protein complex least represented in the active chromatin proteome based on consolidated inputs. '
        'Input: thinking and answers from stage_1.subtask_1 and stage_1.subtask_2.'
    )
    final_decision_instruction_7 = (
        'Sub-task 1: Synthesize and choose the most consistent answer for the least represented protein complex based on consolidated analysis.'
    )
    cot_sc_desc_7 = {
        'instruction': cot_sc_instruction_7,
        'final_decision_instruction': final_decision_instruction_7,
        'input': [taskInfo, results5['thinking'], results5['answer'], results6['thinking'], results6['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results7, log7 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_sc_desc_7,
        n_repeat=self.max_sc
    )
    logs.append(log7)

    review_desc_8 = {
        'instruction': (
            'Sub-task 1: Evaluate the selected protein complex for correctness and consistency with biological knowledge and experimental context. '
            'Input: thinking and answer from stage_2.subtask_1.'
        ),
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results8, log8 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_8
    )
    logs.append(log8)

    reflexion_desc_9 = {
        'instruction': (
            'Sub-task 1: Refine the final answer by consolidating validation feedback and format it according to the query requirements. '
            'Input: thinking and answers from stage_3.subtask_1 and stage_2.subtask_1.'
        ),
        'critic_instruction': (
            'Please review and provide limitations or improvements for the refined final answer.'
        ),
        'input': [taskInfo, results8['thinking'], results8['answer'], results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results9, log9 = await self.reflexion(
        subtask_id='stage_4.subtask_1',
        reflect_desc=reflexion_desc_9,
        n_repeat=self.max_round
    )
    logs.append(log9)

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'])
    return final_answer, logs
