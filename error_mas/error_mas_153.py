async def forward_153(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        'instruction': 'Sub-task stage_0.subtask_1: Extract and summarize molecular ion peaks and isotope pattern from mass spectrometry data. Input: taskInfo containing question and spectral data.',
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        'instruction': 'Sub-task stage_0.subtask_2: Extract and interpret IR spectral peaks to identify functional groups. Input: taskInfo containing question and spectral data.',
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_agent_desc_0_3 = {
        'instruction': 'Sub-task stage_0.subtask_3: Extract and analyze 1H NMR signals to infer proton environments and substitution pattern. Input: taskInfo containing question and spectral data.',
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)

    cot_agent_desc_1_1 = {
        'instruction': 'Sub-task stage_1.subtask_1: Integrate mass spec, IR, and NMR data to propose possible structural features and relationships. Input: results (thinking and answer) from stage_0.subtask_1, stage_0.subtask_2, and stage_0.subtask_3.',
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer'], results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_agent_desc_2_1 = {
        'instruction': 'Sub-task stage_2.subtask_1: Combine all spectral interpretations to produce a consolidated structural suggestion. Input: results (thinking and answer) from stage_1.subtask_1.',
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    cot_agent_desc_3_1 = {
        'instruction': 'Sub-task stage_3.subtask_1: Evaluate candidate structures against spectral data and select the best matching compound. Input: results (thinking and answer) from stage_2.subtask_1.',
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.cot(subtask_id='stage_3.subtask_1', cot_agent_desc=cot_agent_desc_3_1)
    logs.append(log_3_1)

    cot_agent_desc_4_1 = {
        'instruction': 'Sub-task stage_4.subtask_1: Produce a clear, concise final answer naming the most reasonable structural candidate. Input: results (thinking and answer) from stage_3.subtask_1.',
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    final_answer, log_4_1 = await self.answer_generate(subtask_id='stage_4.subtask_1', cot_agent_desc=cot_agent_desc_4_1)
    logs.append(log_4_1)

    return final_answer, logs
