async def forward_24(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_instruction_0_1 = (
            "Stage 0, Sub-task 1: Rewrite the given logarithmic equations into algebraic form and express variables in terms of logarithms. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively. "
            "Context: user query"
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.6,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Stage 0, Sub-task 2: Solve the resulting linear system for the logarithms of x, y, and z. "
            "Input content are results (both thinking and answer) from: previous.stage_0.subtask_1, respectively. "
            "Context: user query"
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Stage 1, Sub-task 1: Compute the value of log2(x^4 y^3 z^2) using the solved logarithms and find its absolute value as a reduced fraction m/n. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively. "
        "Context: user query"
    )
    final_decision_instruction_1_1 = (
        "Stage 1, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for computing the absolute value of log2(x^4 y^3 z^2) as a reduced fraction m/n."
    )
    cot_sc_desc_1_1 = {
        'instruction': cot_sc_instruction_1_1,
        'final_decision_instruction': final_decision_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=3
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Stage 1, Sub-task 2: Calculate and output the sum m + n where m/n is the simplified fraction from the previous subtask. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively. "
        "Context: user query"
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    formatter_instruction_2_1 = (
        "Stage 2, Sub-task 1: Format the final answer m + n as a single integer output. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively. "
        "Context: user query"
    )
    formatter_desc_2_1 = {
        'instruction': formatter_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2'],
        'format': 'short and concise, without explaination'
    }
    results_2_1, log_2_1 = await self.specific_format(
        subtask_id='stage_2.subtask_1',
        formatter_desc=formatter_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
