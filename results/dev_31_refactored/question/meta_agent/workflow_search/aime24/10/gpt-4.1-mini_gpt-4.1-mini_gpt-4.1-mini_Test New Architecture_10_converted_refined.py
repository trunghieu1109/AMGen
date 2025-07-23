async def forward_10(self, taskInfo):
    logs = []
    loop_results = {}
    loop_results['stage_0.subtask_1'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_2'] = {'thinking': [], 'answer': []}
    loop_results['stage_0.subtask_3'] = {'thinking': [], 'answer': []}
    loop_results['stage_1.subtask_1'] = {'thinking': [], 'answer': []}
    loop_results['stage_2.subtask_1'] = {'thinking': [], 'answer': []}

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and formalize the geometric properties of rectangles ABCD and EFGH, "
            "including side lengths, right angles, and orientation assumptions. Ensure clarity on rectangle properties to avoid misinterpretation of shape and scale. "
            "Input content are results (both thinking and answer) from: none."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
        logs.append(log_0_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])

        cot_instruction_0_2 = (
            "Sub-task 2: Analyze the collinearity of points D, E, C, F and the concyclicity of points A, D, H, G to establish geometric constraints. "
            "Explicitly incorporate the failure reason from previous attempt about misapplication of perpendicular bisector logic and ensure correct interpretation of these constraints. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
        logs.append(log_0_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])

        cot_instruction_0_3 = (
            "Sub-task 3: Enumerate and analyze both possible orientations of rectangle EFGH (FG 'up' vs. FG 'down'). "
            "For each orientation, derive the corresponding geometric equations and constraints, avoiding premature elimination of valid cases as happened previously. "
            "This subtask addresses the critical feedback that both orientations must be tested systematically. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2']
        }
        results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
        logs.append(log_0_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])

        cot_instruction_1_1 = (
            "Sub-task 1: Using the constraints and equations from stage_0.subtask_3, solve for the length CE in each orientation case. "
            "Carefully apply geometric and algebraic methods, ensuring no misapplication of perpendicular bisector or midpoint logic as noted in the previous failure. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        )
        cot_agent_desc_1_1 = {
            'instruction': cot_instruction_1_1,
            'input': [taskInfo, results_0_3['thinking'], results_0_3['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
        }
        results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])

        cot_instruction_2_1 = (
            "Sub-task 1: Assess all candidate CE lengths obtained from stage_1.subtask_1 against all original problem constraints, including point ordering on the collinear line (D–E–C–F), concyclicity, and rectangle properties. "
            "Use a systematic and explicit comparison to avoid losing valid solutions, as happened previously. Select the CE length that fully satisfies all conditions. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_1, respectively."
        )
        cot_agent_desc_2_1 = {
            'instruction': cot_instruction_2_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 1 of stage 1', 'answer of subtask 1 of stage 1']
        }
        results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
        logs.append(log_2_1)
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])

    final_answer = await self.make_final_answer(loop_results['stage_2.subtask_1']['thinking'][-1], loop_results['stage_2.subtask_1']['answer'][-1])
    return final_answer, logs
