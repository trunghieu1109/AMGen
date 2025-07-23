async def forward_188(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_1.subtask_1': {'thinking': [], 'answer': []},
        'stage_1.subtask_2': {'thinking': [], 'answer': []},
        'stage_2.subtask_1': {'thinking': [], 'answer': []},
        'stage_2.subtask_2': {'thinking': [], 'answer': []},
        'stage_2.subtask_3': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_0_1 = {
            'instruction': (
                "Sub-task 1: Extract and summarize the given information about each effective particle (Magnon, Skyrmion, Pion, Phonon) "
                "and its relation to spontaneous symmetry breaking. Input: taskInfo containing the question and choices."
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_agent_desc_0_2 = {
            'instruction': (
                "Sub-task 2: Analyze the physical and theoretical context of each particle, focusing on their association with spontaneously-broken symmetries. "
                "Input: taskInfo, all previous thinking and answers from stage_0.subtask_1 and all previous iterations of stage_2.subtask_3."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_2.subtask_3']['thinking'] + loop_results['stage_2.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_2.subtask_3', 'answer of previous stage_2.subtask_3']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_agent_desc_0_3 = {
            'instruction': (
                "Sub-task 3: Identify ambiguities and clarify the meaning of 'associated with spontaneously-broken symmetry' for each particle. "
                "Input: taskInfo, all previous thinking and answers from stage_0.subtask_2 and all previous iterations of stage_1.subtask_1."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_1.subtask_1', 'answer of previous stage_1.subtask_1']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_0_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        aggregate_desc_1_1 = {
            'instruction': (
                "Sub-task 1: Combine the summarized information and analyses from stage_0 to form a consolidated view of each particle's symmetry association. "
                "Input: taskInfo, all thinking and answers from stage_0.subtask_1, stage_0.subtask_2, stage_0.subtask_3, and all previous iterations of stage_1.subtask_1."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'solutions generated from stage_0']
        }
        results_1_1, log_1_1 = await self.aggregate(
            subtask_id='stage_1.subtask_1',
            aggregate_desc=aggregate_desc_1_1
        )
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])
        logs.append(log_1_1)

        cot_agent_desc_1_2 = {
            'instruction': (
                "Sub-task 2: Apply evaluation criteria to the consolidated data to preliminarily classify each particle as associated or not associated with spontaneously-broken symmetry. "
                "Input: taskInfo, all thinking and answers from stage_1.subtask_1 and all previous iterations of stage_2.subtask_1."
            ),
            'input': [taskInfo] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of previous stage_2.subtask_1', 'answer of previous stage_2.subtask_1']
        }
        results_1_2, log_1_2 = await self.cot(
            subtask_id='stage_1.subtask_2',
            cot_agent_desc=cot_agent_desc_1_2
        )
        loop_results['stage_1.subtask_2']['thinking'].append(results_1_2['thinking'])
        loop_results['stage_1.subtask_2']['answer'].append(results_1_2['answer'])
        logs.append(log_1_2)

        review_desc_2_1 = {
            'instruction': (
                "Sub-task 1: Validate the preliminary classifications against theoretical definitions and known physics principles. "
                "Input: taskInfo, all thinking and answers from stage_1.subtask_2 and all previous iterations of stage_2.subtask_1."
            ),
            'input': [taskInfo] + loop_results['stage_1.subtask_2']['thinking'] + loop_results['stage_1.subtask_2']['answer'] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2', 'thinking of previous stage_2.subtask_1', 'answer of previous stage_2.subtask_1']
        }
        results_2_1, log_2_1 = await self.review(
            subtask_id='stage_2.subtask_1',
            review_desc=review_desc_2_1
        )
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])
        logs.append(log_2_1)

        debate_desc_2_2 = {
            'instruction': (
                "Sub-task 2: Select the particle(s) that do not fit the criteria of being associated with spontaneously-broken symmetry. "
                "Input: taskInfo, all thinking and answers from stage_2.subtask_1."
            ),
            'final_decision_instruction': "Sub-task 2: Select the particle(s) not associated with spontaneously-broken symmetry.",
            'input': [taskInfo] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'],
            'temperature': 0.5
        }
        results_2_2, log_2_2 = await self.debate(
            subtask_id='stage_2.subtask_2',
            debate_desc=debate_desc_2_2,
            n_repeat=self.max_round
        )
        loop_results['stage_2.subtask_2']['thinking'].append(results_2_2['thinking'])
        loop_results['stage_2.subtask_2']['answer'].append(results_2_2['answer'])
        logs.append(log_2_2)

        cot_sc_instruction_2_3 = (
            "Sub-task 3: Assess the validity and consistency of the selection to ensure the correct particle is identified. "
            "Input: taskInfo, all thinking and answers from stage_2.subtask_2 and all previous iterations of stage_2.subtask_3."
        )
        final_decision_instruction_2_3 = "Sub-task 3: Synthesize and choose the most consistent answer for the particle not associated with spontaneously-broken symmetry."
        cot_sc_desc_2_3 = {
            'instruction': cot_sc_instruction_2_3,
            'final_decision_instruction': final_decision_instruction_2_3,
            'input': [taskInfo] + loop_results['stage_2.subtask_2']['thinking'] + loop_results['stage_2.subtask_2']['answer'] + loop_results['stage_2.subtask_3']['thinking'] + loop_results['stage_2.subtask_3']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2', 'thinking of previous stage_2.subtask_3', 'answer of previous stage_2.subtask_3']
        }
        results_2_3, log_2_3 = await self.sc_cot(
            subtask_id='stage_2.subtask_3',
            cot_agent_desc=cot_sc_desc_2_3,
            n_repeat=self.max_sc
        )
        loop_results['stage_2.subtask_3']['thinking'].append(results_2_3['thinking'])
        loop_results['stage_2.subtask_3']['answer'].append(results_2_3['answer'])
        logs.append(log_2_3)

    formatter_desc_3_1 = {
        'instruction': (
            "Sub-task 1: Consolidate the validated selection into a clear, concise final answer specifying which particle is not associated with spontaneously-broken symmetry. "
            "Input: taskInfo and the final thinking and answer from stage_2.subtask_3."
        ),
        'input': [taskInfo, loop_results['stage_2.subtask_3']['thinking'][-1], loop_results['stage_2.subtask_3']['answer'][-1]],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_3', 'answer of stage_2.subtask_3'],
        'format': 'short and concise, without explanation'
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id='stage_3.subtask_1',
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
