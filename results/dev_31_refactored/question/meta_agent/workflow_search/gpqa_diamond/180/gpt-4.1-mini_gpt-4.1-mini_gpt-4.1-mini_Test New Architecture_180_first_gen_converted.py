async def forward_180(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize all relevant information from the query about solar neutrino flux, pp-III branch stoppage, energy bands, and assumptions. "
            "Input: taskInfo containing the question and choices."
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
            "Sub-task 2: Identify and categorize the neutrino energy bands and their expected contributions from different pp chain branches, focusing on pp-III neutrinos. "
            "Input: taskInfo and all previous iteration outputs of stage_0.subtask_1 (thinking and answer)."
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
        "Sub-task 1: Analyze the impact of stopping the pp-III branch on neutrino fluxes in the two energy bands and consolidate the expected flux ratio changes. "
        "Input: taskInfo and all iteration outputs of stage_0.subtask_2 (thinking and answer)."
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
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1: Compute the approximate flux ratio Flux(band 1) / Flux(band 2) after pp-III branch stoppage and validate against known solar neutrino spectra. "
        "Input: taskInfo and all iteration outputs of stage_0.subtask_2 and the output of stage_1.subtask_1 (thinking and answer)."
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + [results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_agent_instruction_3_1 = (
        "Sub-task 1: Format the final answer as the approximate flux ratio and select the correct choice from the given options. "
        "Input: taskInfo and the output of stage_2.subtask_1 (thinking and answer)."
    )
    cot_agent_desc_3_1 = {
        'instruction': cot_agent_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id='stage_3.subtask_1',
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
