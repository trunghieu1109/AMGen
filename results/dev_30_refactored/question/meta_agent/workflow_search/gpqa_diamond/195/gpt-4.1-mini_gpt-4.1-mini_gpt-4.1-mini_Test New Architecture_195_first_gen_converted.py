async def forward_195(self, taskInfo):
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
                'Sub-task 1: Extract and summarize the given physical information and problem parameters relevant to the relativistic harmonic oscillator. '
                'Input content: the user query containing the problem statement and candidate formulas.'
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
                'Sub-task 2: Analyze the physical relationships and constraints between mass, spring constant, amplitude, speed of light, and relativistic effects to form intermediate expressions. '
                'Input content: results (thinking and answer) from stage_0.subtask_1 of all previous iterations and current iteration.'
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
                'Sub-task 3: Simplify and refine the intermediate expressions to approach a candidate formula for maximum speed, ensuring relativistic constraints are respected. '
                'Input content: results (thinking and answer) from stage_0.subtask_2 of all previous iterations and current iteration.'
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

        cot_agent_desc_4 = {
            'instruction': (
                'Sub-task 4: Compare the refined intermediate results with the provided candidate formulas to identify structural matches or discrepancies. '
                'Input content: results (thinking and answer) from stage_0.subtask_3 of all previous iterations and current iteration.'
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results4, log4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

    debate_desc_1 = {
        'instruction': (
            'Sub-task 1: Evaluate each candidate formula against the refined intermediate results and physical constraints to select the best matching formula for maximum speed. '
            'Input content: results (thinking and answer) from all iterations of stage_0.subtask_4.'
        ),
        'final_decision_instruction': (
            'Sub-task 1: Select the best candidate formula for maximum speed based on the evaluation.'
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=2
    )
    logs.append(log5)

    review_desc_1 = {
        'instruction': (
            'Sub-task 1: Validate the selected candidate formula for correctness, consistency with relativistic mechanics, and physical plausibility. '
            'Input content: results (thinking and answer) from stage_1.subtask_1.'
        ),
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results6, log6 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
