async def forward_25(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    formatter_instruction = (
        "Stage 0, Sub-task 1: Analyze the properties of the convex equilateral hexagon with parallel opposite sides and characterize the relationship between the hexagon sides and the triangle formed by extensions of AB, CD, and EF. Input content are results (both thinking and answer) from: none."
    )
    formatter_desc = {
        'instruction': formatter_instruction,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query'],
        'format': 'short and concise, without explaination'
    }
    results_0_1, log_0_1 = await self.specific_format(subtask_id='stage_0.subtask_1', formatter_desc=formatter_desc)
    logs.append(log_0_1)

    # stage_1.subtask_1
    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Construct intermediate steps to relate the triangle side lengths (200, 240, 300) to the hexagon side length using geometric transformations and algebraic expressions. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    loop_results = {
        'stage_2.subtask_1': {'thinking': [], 'answer': []},
        'stage_2.subtask_2': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_sc_instruction_2_1 = (
            f"Stage 2, Sub-task 1: Evaluate candidate values for the hexagon side length based on constructed relationships and validate consistency with given triangle side lengths. "
            f"Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1, respectively. Iteration {iteration+1}."
        )
        final_decision_instruction_2_1 = (
            f"Stage 2, Sub-task 1, Final Decision: Synthesize and choose the most consistent candidate hexagon side length for iteration {iteration+1}."
        )
        cot_sc_desc_2_1 = {
            'instruction': cot_sc_instruction_2_1,
            'final_decision_instruction': final_decision_instruction_2_1,
            'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'] + [f'thinking iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_1']['thinking']))] + [f'answer iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_1']['answer']))]
        }
        results_2_1, log_2_1 = await self.sc_cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_sc_desc_2_1, n_repeat=self.max_sc)
        logs.append(log_2_1)
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])

        cot_sc_instruction_2_2 = (
            f"Stage 2, Sub-task 2: Derive the final quantitative value of the hexagon side length by applying geometric constraints and algebraic simplifications. "
            f"Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1 & stage_2.subtask_1, respectively. Iteration {iteration+1}."
        )
        final_decision_instruction_2_2 = (
            f"Stage 2, Sub-task 2, Final Decision: Synthesize and finalize the hexagon side length for iteration {iteration+1}."
        )
        cot_sc_desc_2_2 = {
            'instruction': cot_sc_instruction_2_2,
            'final_decision_instruction': final_decision_instruction_2_2,
            'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer'], results_2_1['thinking'], results_2_1['answer']] + loop_results['stage_2.subtask_2']['thinking'] + loop_results['stage_2.subtask_2']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'] + [f'thinking iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_2']['thinking']))] + [f'answer iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_2']['answer']))]
        }
        results_2_2, log_2_2 = await self.sc_cot(subtask_id='stage_2.subtask_2', cot_agent_desc=cot_sc_desc_2_2, n_repeat=self.max_sc)
        logs.append(log_2_2)
        loop_results['stage_2.subtask_2']['thinking'].append(results_2_2['thinking'])
        loop_results['stage_2.subtask_2']['answer'].append(results_2_2['answer'])

    cot_sc_instruction_3_1 = (
        "Stage 2, Sub-task 1: Evaluate candidate values for the hexagon side length based on constructed relationships and validate consistency with given triangle side lengths. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1, respectively, and all iterations of stage_2 subtasks."
    )
    final_decision_instruction_3_1 = (
        "Stage 2, Sub-task 1, Final Decision: Synthesize and choose the most consistent candidate hexagon side length from all iterations."
    )
    cot_sc_desc_3_1 = {
        'instruction': cot_sc_instruction_3_1,
        'final_decision_instruction': final_decision_instruction_3_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'] + loop_results['stage_2.subtask_2']['thinking'] + loop_results['stage_2.subtask_2']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'] + [f'thinking iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_1']['thinking']))] + [f'answer iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_1']['answer']))] + [f'thinking iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_2']['thinking']))] + [f'answer iteration {i+1}' for i in range(len(loop_results['stage_2.subtask_2']['answer']))]
    }
    results_3_1, log_3_1 = await self.sc_cot(subtask_id='stage_2.subtask_1.final', cot_agent_desc=cot_sc_desc_3_1, n_repeat=self.max_sc)
    logs.append(log_3_1)

    cot_sc_instruction_3_2 = (
        "Stage 2, Sub-task 2: Derive the final quantitative value of the hexagon side length by applying geometric constraints and algebraic simplifications. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1 & stage_2.subtask_1.final, respectively."
    )
    final_decision_instruction_3_2 = (
        "Stage 2, Sub-task 2, Final Decision: Synthesize and finalize the hexagon side length."
    )
    cot_sc_desc_3_2 = {
        'instruction': cot_sc_instruction_3_2,
        'final_decision_instruction': final_decision_instruction_3_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer'], results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_2.subtask_1.final', 'answer of stage_2.subtask_1.final']
    }
    results_3_2, log_3_2 = await self.sc_cot(subtask_id='stage_2.subtask_2.final', cot_agent_desc=cot_sc_desc_3_2, n_repeat=self.max_sc)
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
