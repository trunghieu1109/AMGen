async def forward_29(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        cot_sc_instruction1 = (
            "Stage 0, Sub-task 1: Classify the problem entities: chips, cells, rows, and columns, and analyze the constraints on chip placement and color uniformity per row and column. "
            "Input content is taskInfo."
        )
        final_decision_instruction1 = (
            "Stage 0, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for classification and constraint analysis."
        )
        cot_sc_desc1 = {
            'instruction': cot_sc_instruction1,
            'final_decision_instruction': final_decision_instruction1,
            'input': [taskInfo],
            'temperature': 0.5,
            'context_desc': ['user query'],
        }
        results1, log1 = await self.sc_cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_sc_desc1,
            n_repeat=3
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results1['answer'])
        logs.append(log1)

        cot_sc_instruction2 = (
            "Stage 0, Sub-task 2: Extract and categorize possible configurations of rows and columns based on color assignments and maximality conditions, constructing intermediate combinatorial models. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        final_decision_instruction2 = (
            "Stage 0, Sub-task 2, Final Decision: Synthesize and choose the most consistent answer for configurations and combinatorial models."
        )
        cot_sc_desc2 = {
            'instruction': cot_sc_instruction2,
            'final_decision_instruction': final_decision_instruction2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1'],
        }
        results2, log2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_sc_desc2,
            n_repeat=3
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

    cot_instruction3 = (
        "Stage 1, Sub-task 1: Evaluate candidate row and column color assignments and select those that satisfy the uniformity and maximality conditions. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2'],
    }
    results3, log3 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc3
    )
    logs.append(log3)

    cot_instruction4 = (
        "Stage 2, Sub-task 1: Validate that each candidate configuration meets all problem constraints, including at most one chip per cell, uniform color per row and column, and maximality of chip placement. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_1, respectively."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + [results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
    }
    results4, log4 = await self.cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    debate_instruction5 = (
        "Stage 3, Sub-task 1: Compute the total number of valid chip placement configurations based on validated candidates, applying combinatorial counting under the given constraints. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_1, respectively."
    )
    final_decision_instruction5 = (
        "Stage 3, Sub-task 1, Final Decision: Derive the final count of valid chip placements satisfying all constraints."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + [results3['thinking'], results3['answer']],
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='stage_3.subtask_1',
        debate_desc=debate_desc5,
        n_repeat=2
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
