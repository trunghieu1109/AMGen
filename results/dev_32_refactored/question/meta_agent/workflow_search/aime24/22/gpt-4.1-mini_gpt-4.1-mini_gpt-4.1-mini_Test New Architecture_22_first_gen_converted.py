async def forward_22(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_instruction_0_1 = (
            "Stage 0, Sub-task 1: Generate candidate lists of positive integers summing to 30 with mode 9 and median as a positive integer not in the list. "
            "Input content: taskInfo"
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.7,
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
            "Stage 0, Sub-task 2: Analyze and transform candidate lists to extract intermediate representations such as frequency counts, median candidates, and mode verification. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1"
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
        "Stage 1, Sub-task 1: Combine and summarize candidate list properties to identify consistent sets meeting sum, mode, and median conditions. "
        "Input content: taskInfo, all thinking and answers from stage_0.subtask_2"
    )
    aggregate_desc_1_1 = {
        'instruction': aggregate_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    evaluate_instruction_2_1 = (
        "Stage 2, Sub-task 1: Evaluate consolidated candidates against all problem constraints and select valid lists. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    evaluate_desc_2_1 = {
        'instruction': evaluate_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id='stage_2.subtask_1',
        aggregate_desc=evaluate_desc_2_1
    )
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Stage 3, Sub-task 1: Format the selected valid list and summarize key properties for final result computation. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
    )
    review_desc_3_1 = {
        'instruction': review_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    answergen_instruction_4_1 = (
        "Stage 4, Sub-task 1: Compute the sum of the squares of all items in the validated list. "
        "Input content: taskInfo, thinking and answer from stage_3.subtask_1"
    )
    answergen_desc_4_1 = {
        'instruction': answergen_instruction_4_1,
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_4_1, log_4_1 = await self.answer_generate(
        subtask_id='stage_4.subtask_1',
        cot_agent_desc=answergen_desc_4_1
    )
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1['thinking'], results_4_1['answer'])
    return final_answer, logs
