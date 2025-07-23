async def forward_185(self, taskInfo):
    logs = []
    loop_results = {}

    for iteration in range(3):
        iter_key = f'iteration_{iteration+1}'
        loop_results[iter_key] = {}

        cot_instruction_0_0 = (
            "Sub-task 0: Analyze the starting compound's structure, stereochemistry, and substituents "
            "to establish a clear molecular framework for the Cope rearrangement, with context from taskInfo."
        )
        cot_agent_desc_0_0 = {
            'instruction': cot_instruction_0_0,
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id=f'stage_0_subtask_0_iter_{iteration+1}',
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        loop_results[iter_key]['subtask_0'] = results_0_0

        cot_instruction_0_1 = (
            "Sub-task 1: Apply the Cope rearrangement mechanism to the starting compound to generate possible "
            "intermediate structures and predict connectivity changes, based on output from Sub-task 0."
        )
        cot_sc_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent answer for applying Cope rearrangement.",
            'input': [taskInfo, results_0_0.get('thinking', ''), results_0_0.get('answer', '')],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of subtask 0', 'answer of subtask 0']
        }
        results_0_1, log_0_1 = await self.sc_cot(
            subtask_id=f'stage_0_subtask_1_iter_{iteration+1}',
            cot_agent_desc=cot_sc_desc_0_1,
            n_repeat=self.max_sc
        )
        logs.append(log_0_1)
        loop_results[iter_key]['subtask_1'] = results_0_1

        cot_instruction_0_2 = (
            "Sub-task 2: Interpret the stereochemical and regiochemical outcomes of the rearrangement, "
            "considering the (1S,4R) configuration and vinyl substituent, based on output from Sub-task 1."
        )
        cot_sc_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent stereochemical interpretation.",
            'input': [taskInfo, results_0_1.get('thinking', ''), results_0_1.get('answer', '')],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
        }
        results_0_2, log_0_2 = await self.sc_cot(
            subtask_id=f'stage_0_subtask_2_iter_{iteration+1}',
            cot_agent_desc=cot_sc_desc_0_2,
            n_repeat=self.max_sc
        )
        logs.append(log_0_2)
        loop_results[iter_key]['subtask_2'] = results_0_2

        aggregate_instruction_0_3 = (
            "Sub-task 3: From solutions generated in Subtask 2, aggregate these solutions and return the consistent "
            "and best solution mapping predicted rearranged structures to the given product choices."
        )
        aggregate_desc_0_3 = {
            'instruction': aggregate_instruction_0_3,
            'input': [taskInfo, 
                      results_0_0.get('answer', ''), 
                      results_0_1.get('answer', ''), 
                      results_0_2.get('answer', '')],
            'temperature': 0.0,
            'context_desc': ['user query', 'solutions generated from subtasks 0,1,2']
        }
        results_0_3, log_0_3 = await self.aggregate(
            subtask_id=f'stage_0_subtask_3_iter_{iteration+1}',
            aggregate_desc=aggregate_desc_0_3
        )
        logs.append(log_0_3)
        loop_results[iter_key]['subtask_3'] = results_0_3

        cot_reflect_instruction_0_4 = (
            "Sub-task 4: Refine and consolidate the intermediate reasoning and structural assignments to produce "
            "a provisional set of candidate products with supporting rationale, based on output from Sub-task 3."
        )
        critic_instruction_0_4 = (
            "Please review and provide the limitations of provided solutions of mapping rearranged structures to products."
        )
        cot_reflect_desc_0_4 = {
            'instruction': cot_reflect_instruction_0_4,
            'critic_instruction': critic_instruction_0_4,
            'input': [taskInfo, 
                      results_0_0.get('thinking', ''), results_0_0.get('answer', ''),
                      results_0_1.get('thinking', ''), results_0_1.get('answer', ''),
                      results_0_2.get('thinking', ''), results_0_2.get('answer', ''),
                      results_0_3.get('thinking', ''), results_0_3.get('answer', '')],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of subtask 0', 'answer of subtask 0',
                'thinking of subtask 1', 'answer of subtask 1',
                'thinking of subtask 2', 'answer of subtask 2',
                'thinking of subtask 3', 'answer of subtask 3'
            ]
        }
        results_0_4, log_0_4 = await self.reflexion(
            subtask_id=f'stage_0_subtask_4_iter_{iteration+1}',
            reflect_desc=cot_reflect_desc_0_4,
            n_repeat=self.max_round
        )
        logs.append(log_0_4)
        loop_results[iter_key]['subtask_4'] = results_0_4

    final_candidates = [loop_results[f'iteration_{i+1}']['subtask_4'].get('answer', '') for i in range(3)]

    eval_instruction_1_0 = (
        "Sub-task 0: Evaluate the provisional candidate products against mechanistic criteria, stereochemical "
        "consistency, and product naming to select the most plausible Cope rearrangement product, based on refined candidates from stage 0."
    )
    answer_generate_desc_1_0 = {
        'instruction': eval_instruction_1_0,
        'input': [taskInfo] + final_candidates,
        'temperature': 0.0,
        'context': ['user query', 'refined candidate products from stage 0']
    }
    results_1_0, log_1_0 = await self.answer_generate(
        subtask_id='stage_1_subtask_0',
        cot_agent_desc=answer_generate_desc_1_0
    )
    logs.append(log_1_0)

    final_answer = await self.make_final_answer(
        results_1_0.get('thinking', ''),
        results_1_0.get('answer', '')
    )

    return final_answer, logs
