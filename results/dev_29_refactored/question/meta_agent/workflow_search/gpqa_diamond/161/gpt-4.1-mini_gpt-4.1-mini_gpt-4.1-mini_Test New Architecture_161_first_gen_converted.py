async def forward_161(self, taskInfo):
    logs = []
    stage0_results = {}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and restate the given metric and domain conditions clearly in LaTeX format. "
            "Given the metric ds^2 = 32/(4 - x^2 - y^2) * (dx^2 + dy^2), restate it and the domain where denominator > 0."
        )
        cot_agent_desc_0_0 = {
            'instruction': cot_instruction_0_0,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id=f'stage_0_subtask_0_iter_{iteration}',
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the geometric meaning of the metric and identify the domain of definition (disk radius 2). "
            "Use the output from Sub-task 0."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo, results_0_0['thinking'], results_0_0.get('answer', '')],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 0']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id=f'stage_0_subtask_1_iter_{iteration}',
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Set up the integral expression for the area of the pseudosphere using the given metric. "
            "Use outputs from Sub-task 1."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo, results_0_1['thinking'], results_0_1.get('answer', '')],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id=f'stage_0_subtask_2_iter_{iteration}',
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Evaluate the integral or analyze its convergence to determine if the area is finite or infinite. "
            "Use outputs from Sub-task 2."
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo, results_0_2['thinking'], results_0_2.get('answer', '')],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 2']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id=f'stage_0_subtask_3_iter_{iteration}',
            cot_agent_desc=cot_agent_desc_0_3
        )
        logs.append(log_0_3)

        cot_agent_instruction_0_4 = (
            "Sub-task 4: Interpret the result in the context of the given multiple-choice answers and identify the correct choice. "
            "Use outputs from Sub-task 3."
        )
        cot_agent_desc_0_4 = {
            'instruction': cot_agent_instruction_0_4,
            'input': [taskInfo, results_0_3['thinking'], results_0_3.get('answer', '')],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 3']
        }
        results_0_4, log_0_4 = await self.answer_generate(
            subtask_id=f'stage_0_subtask_4_iter_{iteration}',
            cot_agent_desc=cot_agent_desc_0_4
        )
        logs.append(log_0_4)

        stage0_results[iteration] = {
            'subtask_0': results_0_0,
            'subtask_1': results_0_1,
            'subtask_2': results_0_2,
            'subtask_3': results_0_3,
            'subtask_4': results_0_4
        }

    aggregate_instruction_1_0 = (
        "Sub-task 0: Combine the intermediate results from stage_0 subtasks into a single coherent conclusion about the area. "
        "Aggregate the results from all 3 iterations of stage_0_subtask_4."
    )
    aggregate_desc_1_0 = {
        'instruction': aggregate_instruction_1_0,
        'input': [taskInfo] + [stage0_results[i]['subtask_4'] for i in range(3)],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_0_subtask_4 iterations']
    }
    results_1_0, log_1_0 = await self.aggregate(
        subtask_id='stage_1_subtask_0',
        aggregate_desc=aggregate_desc_1_0
    )
    logs.append(log_1_0)

    cot_instruction_2_0 = (
        "Sub-task 0: Evaluate the consolidated conclusion against the multiple-choice options and select the matching answer. "
        "Use the aggregated conclusion from stage_1_subtask_0."
    )
    cot_agent_desc_2_0 = {
        'instruction': cot_instruction_2_0,
        'input': [taskInfo, results_1_0['thinking'], results_1_0.get('answer', '')],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1_subtask_0']
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id='stage_2_subtask_0',
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    review_instruction_3_0 = (
        "Sub-task 0: Verify the correctness and consistency of the selected answer with the mathematical reasoning and problem context. "
        "Use the selected answer from stage_2_subtask_0."
    )
    review_desc_3_0 = {
        'instruction': review_instruction_3_0,
        'input': [taskInfo, results_2_0['thinking'], results_2_0.get('answer', '')],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2_subtask_0', 'answer of stage_2_subtask_0']
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id='stage_3_subtask_0',
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    formatter_instruction_4_0 = (
        "Sub-task 0: Simplify and format the final answer and reasoning into a clear LaTeX presentation suitable for output. "
        "Use the reviewed answer from stage_3_subtask_0."
    )
    formatter_desc_4_0 = {
        'instruction': formatter_instruction_4_0,
        'input': [taskInfo, results_3_0['thinking'], results_3_0.get('answer', '')],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3_subtask_0', 'answer of stage_3_subtask_0'],
        'format': 'short and concise, without explaination'
    }
    results_4_0, log_4_0 = await self.specific_format(
        subtask_id='stage_4_subtask_0',
        formatter_desc=formatter_desc_4_0
    )
    logs.append(log_4_0)

    final_answer = await self.make_final_answer(results_4_0['thinking'], results_4_0.get('answer', ''))

    return final_answer, logs
