async def forward_16(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_1.subtask_1': {'thinking': [], 'answer': []},
        'stage_2.subtask_1': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and formalize the geometric conditions from the problem, including the perpendicularity IA perpendicular to OI, "
            "and relate the incenter, circumcenter, and vertex A positions. Input content are results (both thinking and answer) from: none. "
            "Context: user query."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
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
            "Sub-task 2: Express the triangle’s side lengths and angles in terms of the given radii and the geometric constraints identified. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1. Context: user query, thinking and answer of stage_0.subtask_1."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_1_1 = (
            "Sub-task 1: Apply the geometric relations and formulas to calculate the value of AB times AC based on the circumradius, inradius, and perpendicularity condition. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2. Context: user query, thinking and answer of stage_0.subtask_2."
        )
        cot_agent_desc_1_1 = {
            'instruction': cot_instruction_1_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id='stage_1.subtask_1',
            cot_agent_desc=cot_agent_desc_1_1
        )
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])
        logs.append(log_1_1)

        cot_instruction_2_1 = (
            "Sub-task 1: Verify the computed product AB times AC against the problem’s geometric constraints and confirm its validity. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 and stage_1.subtask_1. Context: user query, thinking and answer of stage_0.subtask_2, thinking and answer of stage_1.subtask_1."
        )
        cot_agent_desc_2_1 = {
            'instruction': cot_instruction_2_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id='stage_2.subtask_1',
            cot_agent_desc=cot_agent_desc_2_1
        )
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])
        logs.append(log_2_1)

    final_thinking = loop_results['stage_2.subtask_1']['thinking'][-1]
    final_answer = loop_results['stage_2.subtask_1']['answer'][-1]

    final_answer = await self.make_final_answer(final_thinking, final_answer)
    return final_answer, logs
