async def forward_23(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_agent_desc_1 = {
            'instruction': 'Stage 0, Subtask 1: Enumerate all possible digit assignments to the 2x3 grid cells for the problem: Find the number of ways to place a digit in each cell of a 2x3 grid so that the sum of the two numbers formed by reading left to right is 999, and the sum of the three numbers formed by reading top to bottom is 99. Input content: taskInfo.',
            'input': [taskInfo],
            'temperature': 0.7,
            'context_desc': ['user query']
        }
        results1, log1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results1['answer'])
        logs.append(log1)

        cot_agent_desc_2 = {
            'instruction': 'Stage 0, Subtask 2: Apply the horizontal sum constraint (two numbers formed left to right sum to 999) to filter candidates. Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively.',
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results2, log2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_agent_desc_3 = {
            'instruction': 'Stage 0, Subtask 3: Apply the vertical sum constraint (three numbers formed top to bottom sum to 99) to further filter candidates. Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively.',
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results3, log3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

    aggregate_instruction = 'Stage 1, Subtask 1: Combine all candidates passing both constraints into a single consolidated set. Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively.'
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_0.subtask_3']
    }
    results_agg, log_agg = await self.aggregate(subtask_id='stage_1.subtask_1', aggregate_desc=aggregate_desc)
    logs.append(log_agg)

    sc_cot_instruction = 'Stage 2, Subtask 1: Validate consolidated candidates against all problem criteria and select valid solutions. Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively.'
    final_decision_instruction = 'Stage 2, Subtask 1, Final Decision: Select the most consistent valid solutions for the problem.'
    sc_cot_desc = {
        'instruction': sc_cot_instruction,
        'final_decision_instruction': final_decision_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_sc, log_sc = await self.sc_cot(subtask_id='stage_2.subtask_1', cot_agent_desc=sc_cot_desc, n_repeat=3)
    logs.append(log_sc)

    answer_generate_instruction = 'Stage 3, Subtask 1: Count the number of valid solutions and format the final answer. Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively.'
    answer_generate_desc = {
        'instruction': answer_generate_instruction,
        'input': [taskInfo, results_sc['thinking'], results_sc['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_final, log_final = await self.answer_generate(subtask_id='stage_3.subtask_1', cot_agent_desc=answer_generate_desc)
    logs.append(log_final)

    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'])
    return final_answer, logs
