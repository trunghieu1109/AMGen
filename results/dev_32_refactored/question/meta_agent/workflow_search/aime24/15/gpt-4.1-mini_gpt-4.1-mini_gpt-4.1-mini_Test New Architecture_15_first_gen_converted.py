async def forward_15(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_instruction_0_1 = (
            "Stage 0, Sub-task 1: Identify and list all given ownership counts and total residents; "
            "represent the problem parameters systematically. Input content are results (both thinking and answer) from: none."
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
            "Stage 0, Sub-task 2: Formulate initial equations relating the number of residents owning exactly one, two, three, and four items based on given data. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

    aggregate_instruction_1_1 = (
        "Stage 1, Sub-task 1: Combine the formulated equations and given counts to express the total number of residents and ownership distributions in a consolidated form. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    aggregate_desc_1_1 = {
        'instruction': aggregate_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_2_1 = (
        "Stage 2, Sub-task 1: Analyze the consolidated equations to derive relationships between the number of residents owning exactly one, two, three, and four items. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    aggregate_desc_2_1 = {
        'instruction': aggregate_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id='stage_2.subtask_1',
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    aggregate_instruction_3_1 = (
        "Stage 3, Sub-task 1: Identify the variable representing the number of residents owning all four items and isolate it for calculation. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    aggregate_desc_3_1 = {
        'instruction': aggregate_instruction_3_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_3_1, log_3_1 = await self.aggregate(
        subtask_id='stage_3.subtask_1',
        aggregate_desc=aggregate_desc_3_1
    )
    logs.append(log_3_1)

    cot_instruction_4_1 = (
        "Stage 4, Sub-task 1: Evaluate the isolated variable for the number of residents owning all four items to ensure it is consistent and valid within the problem constraints. "
        "Input content are results (both thinking and answer) from: stage_3.subtask_1, respectively."
    )
    cot_agent_desc_4_1 = {
        'instruction': cot_instruction_4_1,
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_4_1, log_4_1 = await self.cot(
        subtask_id='stage_4.subtask_1',
        cot_agent_desc=cot_agent_desc_4_1
    )
    logs.append(log_4_1)

    cot_instruction_5_1 = (
        "Stage 5, Sub-task 1: Calculate the exact number of residents owning all four items using the derived relationships and validated variables. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    cot_agent_desc_5_1 = {
        'instruction': cot_instruction_5_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_5_1, log_5_1 = await self.cot(
        subtask_id='stage_5.subtask_1',
        cot_agent_desc=cot_agent_desc_5_1
    )
    logs.append(log_5_1)

    cot_instruction_6_1 = (
        "Stage 6, Sub-task 1: Apply the calculated value to finalize the answer for the number of residents owning all four items. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1 & stage_5.subtask_1, respectively."
    )
    cot_agent_desc_6_1 = {
        'instruction': cot_instruction_6_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer'], results_5_1['thinking'], results_5_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1', 'thinking of stage_5.subtask_1', 'answer of stage_5.subtask_1']
    }
    results_6_1, log_6_1 = await self.cot(
        subtask_id='stage_6.subtask_1',
        cot_agent_desc=cot_agent_desc_6_1
    )
    logs.append(log_6_1)

    cot_instruction_7_1 = (
        "Stage 7, Sub-task 1: Consolidate and format the final answer clearly, ensuring it directly addresses the query. "
        "Input content are results (both thinking and answer) from: stage_4.subtask_1 & stage_6.subtask_1, respectively."
    )
    cot_agent_desc_7_1 = {
        'instruction': cot_instruction_7_1,
        'input': [taskInfo, results_4_1['thinking'], results_4_1['answer'], results_6_1['thinking'], results_6_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_4.subtask_1', 'answer of stage_4.subtask_1', 'thinking of stage_6.subtask_1', 'answer of stage_6.subtask_1']
    }
    results_7_1, log_7_1 = await self.cot(
        subtask_id='stage_7.subtask_1',
        cot_agent_desc=cot_agent_desc_7_1
    )
    logs.append(log_7_1)

    final_answer = await self.make_final_answer(results_7_1['thinking'], results_7_1['answer'])
    return final_answer, logs
