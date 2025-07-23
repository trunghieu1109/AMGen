async def forward_6(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        'instruction': 'Sub-task 1: Express the surface area and volume constraints in terms of variables x, y, z. Input content: problem query.',
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        'instruction': 'Sub-task 2: Formulate the expression for the squared radius r^2 = (x^2 + y^2 + z^2)/4. Input content: results (thinking and answer) from stage_0.subtask_1.',
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_agent_desc_1_1 = {
        'instruction': 'Sub-task 1: Apply optimization techniques (e.g., Lagrange multipliers) to maximize r^2 under the given constraints. Input content: results (thinking and answer) from stage_0.subtask_2.',
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_agent_desc_1_2 = {
        'instruction': 'Sub-task 2: Simplify and solve resulting equations to find r^2 in reduced fraction form p/q. Input content: results (thinking and answer) from stage_1.subtask_1.',
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    cot_agent_desc_2_1 = {
        'instruction': 'Sub-task 1: Compute p + q from the reduced fraction p/q and format the final answer. Input content: results (thinking and answer) from stage_1.subtask_2.',
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
