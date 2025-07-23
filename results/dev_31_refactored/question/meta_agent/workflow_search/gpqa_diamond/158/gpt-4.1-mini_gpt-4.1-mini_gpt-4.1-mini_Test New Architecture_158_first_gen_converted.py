async def forward_158(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []}
    }
    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Extract and interpret spectral data to estimate the quasar's redshift based on the observed 790 nm peak and flux drop. "
            "Input includes the user query containing spectral info and cosmological context."
        )
        cot_agent_desc_1 = {
            'instruction': cot_instruction_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results1, log1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results1['answer'])
        logs.append(log1)

        cot_instruction_2 = (
            "Sub-task 2: Calculate the comoving distance corresponding to the estimated redshift using the Lambda-CDM cosmological parameters "
            "(H0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe). "
            "Input includes the user query and all previous redshift estimations (thinking and answers) from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_2 = {
            'instruction': cot_instruction_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results2, log2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

    cot_instruction_3 = (
        "Sub-task 1: Combine redshift and comoving distance estimates to produce a consolidated cosmological distance estimate for the quasar. "
        "Input includes the user query and all comoving distance estimations (thinking and answers) from all iterations of stage_0.subtask_2."
    )
    cot_agent_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results3, log3 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_3
    )
    logs.append(log3)

    debate_instruction_4 = (
        "Sub-task 1: Evaluate the consolidated distance estimate against the provided multiple-choice options (6, 7, 8, 9 Gpc) and select the best matching comoving distance. "
        "Input includes the user query and the consolidated estimate (thinking and answer) from stage_1.subtask_1."
    )
    debate_final_decision_4 = (
        "Sub-task 1: Select the best matching comoving distance from the multiple-choice options based on the consolidated estimate."
    )
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'final_decision_instruction': debate_final_decision_4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1'],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id='stage_2.subtask_1',
        debate_desc=debate_desc_4,
        n_repeat=2
    )
    logs.append(log4)

    answergen_instruction_5 = (
        "Sub-task 1: Format the selected comoving distance into the required final answer format. "
        "Input includes the user query and the selected answer (thinking and answer) from stage_2.subtask_1."
    )
    answergen_desc_5 = {
        'instruction': answergen_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results5, log5 = await self.answer_generate(
        subtask_id='stage_3.subtask_1',
        cot_agent_desc=answergen_desc_5
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
