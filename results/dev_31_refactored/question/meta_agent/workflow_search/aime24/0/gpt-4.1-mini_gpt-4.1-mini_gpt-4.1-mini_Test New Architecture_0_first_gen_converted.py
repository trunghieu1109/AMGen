async def forward_0(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        'instruction': 'Sub-task 1: Define variables s and t and express total times as equations involving walking speed s, coffee time t (in minutes), and given total times (4 hours and 2 hours 24 minutes). Input content: problem query from taskInfo.',
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        'instruction': 'Sub-task 2: Set up the system of equations using the two scenarios and the constant coffee time t. Input content: results (thinking and answer) from stage_0.subtask_1.',
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_agent_desc_1_1 = {
        'instruction': 'Sub-task 1: Solve the system of equations to find values of s and t. Input content: results (thinking and answer) from stage_0.subtask_2.',
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_agent_desc_1_2 = {
        'instruction': 'Sub-task 2: Calculate the total time (walking + coffee) when walking at speed s + 0.5 km/h using the solved values of s and t. Input content: results (thinking and answer) from stage_1.subtask_1.',
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.cot(subtask_id='stage_1.subtask_2', cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    cot_agent_desc_2_1 = {
        'instruction': 'Sub-task 1: Convert the total time from hours to minutes and format the final answer clearly. Input content: results (thinking and answer) from stage_1.subtask_2.',
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
