async def forward_154(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Normalize the given state vector and verify it is an eigenstate of Px with eigenvalue -hbar. "
            "Input content are results (both thinking and answer) from: none. "
            "Input: taskInfo containing matrices Px, Py, Pz and the given state vector."
        )
        cot_agent_desc_1 = {
            'instruction': cot_instruction_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_1, log_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_instruction_2 = (
            "Sub-task 2: Calculate the expectation value <Pz> in the given state using the Pz matrix and normalized state vector. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively. "
            "Input: taskInfo and all previous thinking and answers from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_2 = {
            'instruction': cot_instruction_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_2, log_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_instruction_3 = (
            "Sub-task 3: Calculate the expectation value <Pz^2> in the given state using the square of the Pz matrix and normalized state vector. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively. "
            "Input: taskInfo and all previous thinking and answers from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_3 = {
            'instruction': cot_instruction_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_3, log_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        cot_instruction_4 = (
            "Sub-task 4: Compute the variance (Delta Pz)^2 = <Pz^2> - <Pz>^2 and then the uncertainty Delta Pz by taking the square root. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_0.subtask_3, respectively. "
            "Input: all previous thinking and answers from stage_0.subtask_2 and stage_0.subtask_3 iterations."
        )
        cot_agent_desc_4 = {
            'instruction': cot_instruction_4,
            'input': (
                [taskInfo] +
                loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] +
                loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer']
            ),
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2',
                'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3'
            ]
        }
        results_4, log_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        cot_reflect_instruction_5 = (
            "Sub-task 5: Refine and simplify the expression for Delta Pz to match one of the given multiple-choice options. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4 & former iterations of stage_0.subtask_5, respectively. "
            "Input: all previous thinking and answers from stage_0.subtask_4 and stage_0.subtask_5 iterations."
        )
        critic_instruction_5 = (
            "Please review and provide the limitations of provided solutions of Delta Pz refinement and simplification."
        )
        cot_reflect_desc_5 = {
            'instruction': cot_reflect_instruction_5,
            'critic_instruction': critic_instruction_5,
            'input': (
                [taskInfo] +
                loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'] +
                loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer']
            ),
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4',
                'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'
            ]
        }
        results_5, log_5 = await self.reflexion(
            subtask_id='stage_0.subtask_5',
            reflect_desc=cot_reflect_desc_5,
            n_repeat=1
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

    debate_instruction_1 = (
        "Stage 1 Sub-task 1: Evaluate the refined uncertainty Delta Pz against the provided choices and select the best matching candidate. "
        "Input content are results (both thinking and answer) from: all iterations of stage_0.subtask_5. "
        "Input: taskInfo and all thinking and answers from stage_0.subtask_5 iterations."
    )
    final_decision_instruction_1 = (
        "Stage 1 Sub-task 1: Select the best matching uncertainty Delta Pz from the refined results."
    )
    debate_desc_1 = {
        'instruction': debate_instruction_1,
        'final_decision_instruction': final_decision_instruction_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5'],
        'temperature': 0.5
    }
    results_6, log_6 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=1
    )
    logs.append(log_6)

    final_answer = await self.make_final_answer(results_6['thinking'], results_6['answer'])
    return final_answer, logs
