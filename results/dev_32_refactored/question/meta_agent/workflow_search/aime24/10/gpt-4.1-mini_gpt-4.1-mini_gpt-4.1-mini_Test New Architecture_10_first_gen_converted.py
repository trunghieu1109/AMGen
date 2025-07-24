async def forward_10(self, taskInfo):
    logs = []
    results = {}

    # Stage 0, Sub-task 1 (1)
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Identify and express given geometric conditions (collinearity of D,E,C,F and concyclicity of A,D,H,G) and known side lengths, simplifying the problem setup. "
        "Input content are results (both thinking and answer) from: none."
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)
    results['stage_0.subtask_1'] = results_0_1

    # Stage 1, Sub-task 1 (2)
    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Analyze the implications of the collinearity and concyclicity conditions on the rectangles and their sides, and relate given lengths to unknown segment CE. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)
    results['stage_1.subtask_1'] = results_1_1

    loop_results = {
        'stage_2.subtask_1': {'thinking': [], 'answer': []},
        'stage_3.subtask_1': {'thinking': [], 'answer': []},
        'stage_4.subtask_1': {'thinking': [], 'answer': []}
    }

    for iteration in range(3):
        # Stage 2, Sub-task 1 (3)
        cot_instruction_2_1 = (
            "Stage 2, Sub-task 1: Generate intermediate geometric relations and expressions involving CE using the given conditions and prior analysis. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1 & former iterations of stage_4.subtask_1, respectively."
        )
        input_2_1 = [taskInfo, results_1_1['thinking'], results_1_1['answer']]
        if iteration > 0:
            input_2_1 += loop_results['stage_4.subtask_1']['thinking'] + loop_results['stage_4.subtask_1']['answer']
        cot_agent_desc_2_1 = {
            'instruction': cot_instruction_2_1,
            'input': input_2_1,
            'temperature': 0.7,
            'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'] + (['thinking and answer of previous stage_4.subtask_1 iterations'] if iteration > 0 else [])
        }
        results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
        logs.append(log_2_1)
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])

        # Stage 3, Sub-task 1 (4)
        cot_instruction_3_1 = (
            "Stage 3, Sub-task 1: Apply geometric theorems and algebraic manipulations to simplify and solve for CE based on intermediate relations. "
            "Input content are results (both thinking and answer) from: stage_2.subtask_1 & former iterations of stage_4.subtask_1, respectively."
        )
        input_3_1 = [taskInfo, results_2_1['thinking'], results_2_1['answer']]
        if iteration > 0:
            input_3_1 += loop_results['stage_4.subtask_1']['thinking'] + loop_results['stage_4.subtask_1']['answer']
        cot_agent_desc_3_1 = {
            'instruction': cot_instruction_3_1,
            'input': input_3_1,
            'temperature': 0.7,
            'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'] + (['thinking and answer of previous stage_4.subtask_1 iterations'] if iteration > 0 else [])
        }
        results_3_1, log_3_1 = await self.cot(subtask_id='stage_3.subtask_1', cot_agent_desc=cot_agent_desc_3_1)
        logs.append(log_3_1)
        loop_results['stage_3.subtask_1']['thinking'].append(results_3_1['thinking'])
        loop_results['stage_3.subtask_1']['answer'].append(results_3_1['answer'])

        # Stage 4, Sub-task 1 (5)
        cot_reflect_instruction_4_1 = (
            "Stage 4, Sub-task 1: Evaluate the validity and consistency of derived expressions and partial solutions for CE, providing feedback for refinement. "
            "Input content are results (both thinking and answer) from: stage_3.subtask_1, respectively."
        )
        critic_instruction_4_1 = (
            "Stage 4, Sub-task 1, Criticism: Please review and provide the limitations of provided solutions of CE from stage_3.subtask_1."
        )
        input_4_1 = [taskInfo, results_3_1['thinking'], results_3_1['answer']]
        cot_reflect_desc_4_1 = {
            'instruction': cot_reflect_instruction_4_1,
            'critic_instruction': critic_instruction_4_1,
            'input': input_4_1,
            'temperature': 0.6,
            'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
        }
        results_4_1, log_4_1 = await self.reflexion(subtask_id='stage_4.subtask_1', reflect_desc=cot_reflect_desc_4_1, n_repeat=1)
        logs.append(log_4_1)
        loop_results['stage_4.subtask_1']['thinking'].append(results_4_1['thinking'])
        loop_results['stage_4.subtask_1']['answer'].append(results_4_1['answer'])

    # Stage 5, Sub-task 1 (6)
    cot_agent_instruction_5_1 = (
        "Stage 5, Sub-task 1: Consolidate all prior results and select the most consistent and valid length for CE as the final answer. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1 & stage_2.subtask_1 & stage_3.subtask_1 & stage_4.subtask_1, respectively."
    )
    input_5_1 = [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']]
    input_5_1 += loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer']
    input_5_1 += loop_results['stage_3.subtask_1']['thinking'] + loop_results['stage_3.subtask_1']['answer']
    input_5_1 += loop_results['stage_4.subtask_1']['thinking'] + loop_results['stage_4.subtask_1']['answer']
    cot_agent_desc_5_1 = {
        'instruction': cot_agent_instruction_5_1,
        'input': input_5_1,
        'temperature': 0.0,
        'context_desc': [
            'user query',
            'thinking and answer of stage_0.subtask_1',
            'thinking and answer of stage_1.subtask_1',
            'thinking and answer of stage_2.subtask_1 iterations',
            'thinking and answer of stage_3.subtask_1 iterations',
            'thinking and answer of stage_4.subtask_1 iterations'
        ]
    }
    results_5_1, log_5_1 = await self.answer_generate(subtask_id='stage_5.subtask_1', cot_agent_desc=cot_agent_desc_5_1)
    logs.append(log_5_1)

    final_answer = await self.make_final_answer(results_5_1['thinking'], results_5_1['answer'])
    return final_answer, logs
