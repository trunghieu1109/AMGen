async def forward_172(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                "Sub-task 1: Extract and summarize all given physical quantities and constants relevant to the problem, "
                "including electron speed v, position uncertainty Δx, electron mass m, and reduced Planck constant ħ. "
                "Input content: user query provided in taskInfo."
            ),
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

        cot_agent_desc_2 = {
            'instruction': (
                "Sub-task 2: Apply the Heisenberg uncertainty principle to calculate the minimum uncertainty in momentum Δp "
                "using Δx and ħ. Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_0.subtask_2."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_0.subtask_2', 'answer of previous stage_0.subtask_2']
        }
        results2, log2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_agent_desc_3 = {
            'instruction': (
                "Sub-task 3: Calculate the minimum uncertainty in kinetic energy ΔE from the uncertainty in momentum Δp "
                "and electron mass m, using the relation E = p²/(2m). Input content are results (both thinking and answer) from: "
                "stage_0.subtask_2 & stage_0.subtask_1 & former iterations of stage_0.subtask_3."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_3']['thinking'],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_2',
                'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_1',
                'answer of previous stage_0.subtask_3', 'thinking of previous stage_0.subtask_3'
            ]
        }
        results3, log3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

        cot_reflect_desc_4 = {
            'instruction': (
                "Sub-task 4: Refine and consolidate the intermediate results to produce a clear, simplified estimate of the minimum energy uncertainty ΔE, "
                "ensuring units and magnitudes are consistent. Input content are results (both thinking and answer) from: "
                "stage_0.subtask_3 & former iterations of stage_0.subtask_4."
            ),
            'critic_instruction': (
                "Please review and provide the limitations of provided solutions of Sub-task 4, focusing on clarity, unit consistency, and magnitude correctness."
            ),
            'input': [
                taskInfo,
                *loop_results['stage_0.subtask_3']['thinking'],
                *loop_results['stage_0.subtask_3']['answer'],
                *loop_results['stage_0.subtask_4']['thinking'],
                *loop_results['stage_0.subtask_4']['answer']
            ],
            'temperature': 0.0,
            'context_desc': [
                'user query',
                'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3',
                'thinking of previous stage_0.subtask_4', 'answer of previous stage_0.subtask_4'
            ]
        }
        results4, log4 = await self.reflexion(
            subtask_id='stage_0.subtask_4',
            reflect_desc=cot_reflect_desc_4,
            n_repeat=1
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

    cot_debate_desc_1 = {
        'instruction': (
            "Sub-task 1: Compare the refined estimate of ΔE from stage_0 with the provided multiple-choice options and select the choice that best matches the calculated value. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4."
        ),
        'final_decision_instruction': (
            "Sub-task 1: Synthesize and select the best matching multiple-choice answer for the minimum uncertainty in energy ΔE."
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4'],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=cot_debate_desc_1,
        n_repeat=1
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
