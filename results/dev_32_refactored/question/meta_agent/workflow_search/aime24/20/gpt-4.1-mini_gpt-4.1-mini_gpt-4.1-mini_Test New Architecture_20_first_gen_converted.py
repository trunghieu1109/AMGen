async def forward_20(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []}
    }

    for i in range(2, 5):  # Loop 3 iterations for bases 2,3,4
        b = i

        # stage_0.subtask_1
        cot_instruction_01 = (
            f"Stage 0, Sub-task 1: For the current base b={b}, generate all two-digit numbers in base b and compute their decimal values and digit sums. "
            "Input content: taskInfo"
        )
        cot_agent_desc_01 = {
            'instruction': cot_instruction_01,
            'input': [taskInfo, str(b)],
            'temperature': 0.6,
            'context_desc': ['user query']
        }
        results_01, log_01 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_01
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_01['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_01['answer'])
        logs.append(log_01)

        # stage_0.subtask_2
        cot_instruction_02 = (
            f"Stage 0, Sub-task 2: Identify which generated numbers satisfy the b-eautiful condition for base b={b}: digit sum equals the square root of the number. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 of current iteration"
        )
        cot_agent_desc_02 = {
            'instruction': cot_instruction_02,
            'input': [taskInfo, results_01['thinking'], results_01['answer']],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_02, log_02 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_02
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_02['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_02['answer'])
        logs.append(log_02)

        # stage_0.subtask_3
        cot_instruction_03 = (
            f"Stage 0, Sub-task 3: Count the number of b-eautiful integers found for base b={b} and refine the candidate base if needed. "
            "Input content: results (thinking and answer) from stage_0.subtask_2 of current iteration"
        )
        cot_agent_desc_03 = {
            'instruction': cot_instruction_03,
            'input': [taskInfo, results_02['thinking'], results_02['answer']],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_03, log_03 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_03
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_03['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_03['answer'])
        logs.append(log_03)

    # stage_1.subtask_1
    cot_instruction_11 = (
        "Stage 1, Sub-task 1: Evaluate counts of b-eautiful integers from all iterations and determine the smallest base b≥2 with more than ten such integers. "
        "Input content: results (thinking and answer) from all iterations of stage_0.subtask_3"
    )
    cot_agent_desc_11 = {
        'instruction': cot_instruction_11,
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_11, log_11 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_11
    )
    logs.append(log_11)

    # stage_1.subtask_2
    cot_agent_instruction_12 = (
        "Stage 1, Sub-task 2: Validate the selected base and finalize the answer. "
        "Input content: results (thinking and answer) from stage_1.subtask_1"
    )
    cot_agent_desc_12 = {
        'instruction': cot_agent_instruction_12,
        'input': [taskInfo, results_11['thinking'], results_11['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_12, log_12 = await self.answer_generate(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_12
    )
    logs.append(log_12)

    final_answer = await self.make_final_answer(results_12['thinking'], results_12['answer'])
    return final_answer, logs
