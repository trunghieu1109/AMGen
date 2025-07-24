async def forward_14(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_instruction_0_1 = (
            "Stage 0, Sub-task 1: Parameterize points A, B, C, D on the hyperbola and express rhombus conditions with diagonals intersecting at the origin. "
            "Input content: taskInfo"
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'final_decision_instruction': "Stage 0, Sub-task 1, Final Decision: Synthesize and choose the most consistent parameterization and conditions for points A, B, C, D on the hyperbola forming a rhombus with diagonals intersecting at the origin.",
            'input': [taskInfo],
            'temperature': 0.6,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.sc_cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1,
            n_repeat=3
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Stage 0, Sub-task 2: Formulate algebraic expressions for the diagonals and rhombus side length constraints. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 of current iteration"
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'final_decision_instruction': "Stage 0, Sub-task 2, Final Decision: Synthesize and choose the most consistent algebraic expressions for diagonals and side length constraints based on stage_0.subtask_1 outputs.",
            'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            'temperature': 0.7,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2,
            n_repeat=3
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Combine the parameterized conditions into a single system describing all possible rhombi on the hyperbola. "
        "Input content: results (thinking and answer) from all iterations of stage_0.subtask_1 and stage_0.subtask_2"
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1 all iterations', 'answer of stage_0.subtask_1 all iterations', 'thinking of stage_0.subtask_2 all iterations', 'answer of stage_0.subtask_2 all iterations']
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_2_1 = (
        "Stage 2, Sub-task 1: Validate the system for geometric feasibility and select valid rhombus configurations satisfying all constraints. "
        "Input content: results (thinking and answer) from stage_1.subtask_1"
    )
    aggregate_desc_2_1 = {
        'instruction': aggregate_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id='stage_2.subtask_1',
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_3_1 = (
        "Stage 3, Sub-task 1: Analyze the relationship between the diagonals and side lengths to express BD^2 in terms of parameters. "
        "Input content: results (thinking and answer) from stage_1.subtask_1"
    )
    cot_agent_desc_3_1 = {
        'instruction': cot_instruction_3_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id='stage_3.subtask_1',
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    cot_sc_instruction_4_1 = (
        "Stage 4, Sub-task 1: Derive the greatest real number less than BD^2 valid for all such rhombi using the analyzed relationships and constraints. "
        "Input content: results (thinking and answer) from stage_2.subtask_1 and stage_3.subtask_1"
    )
    final_decision_instruction_4_1 = (
        "Stage 4, Sub-task 1, Final Decision: Synthesize and choose the most consistent greatest real number less than BD^2 valid for all rhombi."
    )
    cot_sc_desc_4_1 = {
        'instruction': cot_sc_instruction_4_1,
        'final_decision_instruction': final_decision_instruction_4_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer'], results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_4_1, log_4_1 = await self.sc_cot(
        subtask_id='stage_4.subtask_1',
        cot_agent_desc=cot_sc_desc_4_1,
        n_repeat=3
    )
    logs.append(log_4_1)

    cot_instruction_5_1 = (
        "Stage 5, Sub-task 1: Compute the final value of the greatest real number less than BD^2 applying all derived constraints. "
        "Input content: results (thinking and answer) from stage_3.subtask_1 and stage_4.subtask_1"
    )
    cot_agent_desc_5_1 = {
        'instruction': cot_instruction_5_1,
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer'], results_4_1['thinking'], results_4_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1', 'thinking of stage_4.subtask_1', 'answer of stage_4.subtask_1']
    }
    results_5_1, log_5_1 = await self.cot(
        subtask_id='stage_5.subtask_1',
        cot_agent_desc=cot_agent_desc_5_1
    )
    logs.append(log_5_1)

    formatter_instruction_6_1 = (
        "Stage 6, Sub-task 1: Format and summarize the final answer clearly and coherently. "
        "Input content: results (thinking and answer) from stage_5.subtask_1"
    )
    formatter_desc_6_1 = {
        'instruction': formatter_instruction_6_1,
        'input': [taskInfo, results_5_1['thinking'], results_5_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_5.subtask_1', 'answer of stage_5.subtask_1'],
        'format': 'short and concise, without explanation'
    }
    results_6_1, log_6_1 = await self.specific_format(
        subtask_id='stage_6.subtask_1',
        formatter_desc=formatter_desc_6_1
    )
    logs.append(log_6_1)

    final_answer = await self.make_final_answer(results_6_1['thinking'], results_6_1['answer'])
    return final_answer, logs
