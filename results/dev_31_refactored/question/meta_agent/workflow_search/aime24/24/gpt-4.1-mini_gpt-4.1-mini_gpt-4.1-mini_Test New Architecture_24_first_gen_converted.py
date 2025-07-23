async def forward_24(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        'instruction': 'Sub-task 1: Rewrite each logarithmic equation in terms of variables a = log2(x), b = log2(y), c = log2(z) to form a linear system. Input content: Query with equations log2(x/(yz))=1/2, log2(y/(xz))=1/3, log2(z/(xy))=1/4, and variables x,y,z positive real numbers.',
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        'instruction': 'Sub-task 2: Express the target expression log2(x^4 y^3 z^2) as a linear combination of a, b, and c. Input content: results (thinking and answer) from stage_0.subtask_1.',
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_agent_desc_1_1 = {
        'instruction': 'Sub-task 1: Solve the linear system for a, b, and c using the transformed equations from stage_0.subtask_1. Input content: results (thinking and answer) from stage_0.subtask_1.',
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_agent_desc_1_2 = {
        'instruction': 'Sub-task 2: Calculate the value of log2(x^4 y^3 z^2) using the solved values of a, b, and c. Input content: results (thinking and answer) from stage_0.subtask_2 and stage_1.subtask_1.',
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    formatter_desc_2_1 = {
        'instruction': 'Sub-task 1: Simplify the absolute value of the computed logarithm to a reduced fraction m/n and compute m+n. Input content: results (thinking and answer) from stage_1.subtask_2.',
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2'],
        'format': 'short and concise, without explanation'
    }
    results_2_1, log_2_1 = await self.specific_format(subtask_id='stage_2.subtask_1', formatter_desc=formatter_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
