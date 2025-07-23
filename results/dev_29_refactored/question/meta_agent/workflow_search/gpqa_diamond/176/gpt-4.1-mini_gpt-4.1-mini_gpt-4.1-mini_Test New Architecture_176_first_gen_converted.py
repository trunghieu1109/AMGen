async def forward_176(self, taskInfo):
    logs = []
    loop_results = {}

    for iteration in range(3):
        iteration_key = f'iteration_{iteration+1}'
        loop_results[iteration_key] = {}

        cot_instruction0 = "Sub-task 0: Extract and summarize all given information from the query to establish a clear data foundation."
        cot_agent_desc0 = {
            'instruction': cot_instruction0,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results0, log0 = await self.cot(
            subtask_id=f'stage_0_subtask_0_iter_{iteration+1}',
            cot_agent_desc=cot_agent_desc0
        )
        logs.append(log0)
        loop_results[iteration_key]['subtask_0'] = results0

        cot_instruction1 = "Sub-task 1: Analyze the physical relationships between radius, mass, temperature, and luminosity based on black body radiation and Wien's law."
        cot_agent_desc1 = {
            'instruction': cot_instruction1,
            'input': [taskInfo, results0['thinking'], results0['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 0', 'answer of subtask 0']
        }
        results1, log1 = await self.cot(
            subtask_id=f'stage_0_subtask_1_iter_{iteration+1}',
            cot_agent_desc=cot_agent_desc1
        )
        logs.append(log1)
        loop_results[iteration_key]['subtask_1'] = results1

        cot_instruction2 = "Sub-task 2: Calculate the temperature of both stars using the peak wavelength and confirm they are equal."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results2, log2 = await self.cot(
            subtask_id=f'stage_0_subtask_2_iter_{iteration+1}',
            cot_agent_desc=cot_agent_desc2
        )
        logs.append(log2)
        loop_results[iteration_key]['subtask_2'] = results2

        cot_instruction3 = "Sub-task 3: Determine the luminosity ratio of Star_1 to Star_2 using the radius ratio and temperature equality."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2']
        }
        results3, log3 = await self.cot(
            subtask_id=f'stage_0_subtask_3_iter_{iteration+1}',
            cot_agent_desc=cot_agent_desc3
        )
        logs.append(log3)
        loop_results[iteration_key]['subtask_3'] = results3

        cot_reflect_instruction4 = "Sub-task 4: Refine and consolidate the intermediate results to produce a clear, simplified luminosity ratio value."
        critic_instruction4 = "Please review and provide the limitations of provided solutions of luminosity ratio calculation and suggest improvements if any."
        cot_reflect_desc4 = {
            'instruction': cot_reflect_instruction4,
            'critic_instruction': critic_instruction4,
            'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of subtask 0', 'answer of subtask 0', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 3', 'answer of subtask 3']
        }
        results4, log4 = await self.reflexion(
            subtask_id=f'stage_0_subtask_4_iter_{iteration+1}',
            reflect_desc=cot_reflect_desc4,
            n_repeat=self.max_round
        )
        logs.append(log4)
        loop_results[iteration_key]['subtask_4'] = results4

    last_iter = loop_results['iteration_3']

    cot_agent_instruction = "Sub-task 0: Compare the calculated luminosity ratio with the provided multiple-choice options to select the best matching candidate."
    cot_agent_desc = {
        'instruction': cot_agent_instruction,
        'input': [taskInfo, last_iter['subtask_4']['thinking'], last_iter['subtask_4']['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0_subtask_4', 'answer of stage_0_subtask_4']
    }
    results_final, log_final = await self.answer_generate(
        subtask_id='stage_1_subtask_0',
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log_final)

    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'])

    return final_answer, logs
