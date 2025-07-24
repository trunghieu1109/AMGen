async def forward_6(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_instruction_01 = (
            "Stage 0, Sub-task 1: Parameterize the rectangular box dimensions (x, y, z) and express the surface area and volume constraints algebraically. "
            "Input content: [taskInfo]."
        )
        cot_agent_desc_01 = {
            'instruction': cot_instruction_01,
            'input': [taskInfo],
            'temperature': 0.6,
            'context_desc': ['user query']
        }
        results_01, log_01 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_01
        )
        logs.append(log_01)
        loop_results['stage_0.subtask_1']['thinking'].append(results_01['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_01['answer'])

        cot_instruction_02 = (
            "Stage 0, Sub-task 2: Derive an expression for the radius squared of the smallest sphere containing the box in terms of x, y, z. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_0.subtask_3."
        )
        cot_agent_desc_02 = {
            'instruction': cot_instruction_02,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_3', 'answer of previous stage_0.subtask_3']
        }
        results_02, log_02 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_02
        )
        logs.append(log_02)
        loop_results['stage_0.subtask_2']['thinking'].append(results_02['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_02['answer'])

        cot_instruction_03 = (
            "Stage 0, Sub-task 3: Iteratively analyze and simplify the constraints to identify the box dimensions that minimize the radius squared, "
            "using prior iteration results. Input content: results (thinking and answer) from stage_0.subtask_1 and stage_0.subtask_2."
        )
        cot_agent_desc_03 = {
            'instruction': cot_instruction_03,
            'input': [taskInfo, results_01['thinking'], results_01['answer'], results_02['thinking'], results_02['answer']],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_03, log_03 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_03
        )
        logs.append(log_03)
        loop_results['stage_0.subtask_3']['thinking'].append(results_03['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_03['answer'])

    cot_instruction_11 = (
        "Stage 1, Sub-task 1: Consolidate the iterative results to find the exact simplified fraction form of r^2 and verify the solution satisfies all constraints. "
        "Input content: results (thinking and answer) from stage_0.subtask_3 from all iterations."
    )
    cot_agent_desc_11 = {
        'instruction': cot_instruction_11,
        'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of all stage_0.subtask_3 iterations', 'answer of all stage_0.subtask_3 iterations']
    }
    results_11, log_11 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_11
    )
    logs.append(log_11)

    cot_agent_instruction_21 = (
        "Stage 2, Sub-task 1: Express r^2 as a reduced fraction p/q and compute p+q for the final answer. "
        "Input content: results (thinking and answer) from stage_1.subtask_1."
    )
    cot_agent_desc_21 = {
        'instruction': cot_agent_instruction_21,
        'input': [taskInfo, results_11['thinking'], results_11['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_21, log_21 = await self.answer_generate(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_agent_desc_21
    )
    logs.append(log_21)

    final_answer = await self.make_final_answer(results_21['thinking'], results_21['answer'])
    return final_answer, logs
