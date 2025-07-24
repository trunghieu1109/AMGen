async def forward_12(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for iteration in range(3):
        cot_sc_instruction_1 = (
            "Stage 0, Sub-task 1: Express the given expression in terms of z = 4e^{iθ} and rewrite the expression to isolate the real part as a function of θ. "
            "Input content are from: taskInfo."
        )
        final_decision_instruction_1 = (
            "Stage 0, Sub-task 1, Final Decision: Synthesize and choose the most consistent expression form for the real part as a function of θ."
        )
        cot_sc_desc_1 = {
            'instruction': cot_sc_instruction_1,
            'final_decision_instruction': final_decision_instruction_1,
            'input': [taskInfo],
            'temperature': 0.5,
            'context_desc': ['user query'],
        }
        results_1, log_1 = await self.sc_cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_sc_desc_1,
            n_repeat=3
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_sc_instruction_2 = (
            "Stage 0, Sub-task 2: Simplify and transform the expression to a form suitable for maximization, e.g., using trigonometric identities. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        final_decision_instruction_2 = (
            "Stage 0, Sub-task 2, Final Decision: Synthesize and choose the most consistent simplified expression suitable for maximization."
        )
        cot_sc_desc_2 = {
            'instruction': cot_sc_instruction_2,
            'final_decision_instruction': final_decision_instruction_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1'],
        }
        results_2, log_2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_sc_desc_2,
            n_repeat=3
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

    cot_instruction_3 = (
        "Stage 1, Sub-task 1: Determine critical points or candidate θ values that maximize the real part based on the simplified expression. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2'],
    }
    results_3, log_3 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_3
    )
    logs.append(log_3)

    cot_sc_instruction_4 = (
        "Stage 2, Sub-task 1: Evaluate the real part at candidate θ values and select the largest value as the final answer. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_1, respectively."
    )
    final_decision_instruction_4 = (
        "Stage 2, Sub-task 1, Final Decision: Synthesize and choose the maximum real part value as the final answer."
    )
    cot_sc_desc_4 = {
        'instruction': cot_sc_instruction_4,
        'final_decision_instruction': final_decision_instruction_4,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + [results_3['thinking'], results_3['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
    }
    results_4, log_4 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_sc_desc_4,
        n_repeat=3
    )
    logs.append(log_4)

    final_answer = await self.make_final_answer(results_4['thinking'], results_4['answer'])
    return final_answer, logs
