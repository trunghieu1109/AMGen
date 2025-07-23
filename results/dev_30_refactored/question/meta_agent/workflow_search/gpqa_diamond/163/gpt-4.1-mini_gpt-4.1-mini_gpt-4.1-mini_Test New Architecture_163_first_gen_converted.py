async def forward_163(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize all given observational data (periods and radial velocity amplitudes) "
            "for system_1 and system_2 into structured physical parameters. Input content is the user query from taskInfo."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_reflect_instruction_0_2 = (
            "Sub-task 2: Calculate the mass ratios of the two stars in each system using the inverse ratio of their radial velocity amplitudes. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & former iterations of stage_0.subtask_5, respectively."
        )
        critic_instruction_0_2 = (
            "Please review and provide the limitations of provided solutions of mass ratio calculations from previous iterations."
        )
        cot_reflect_desc_0_2 = {
            'instruction': cot_reflect_instruction_0_2,
            'critic_instruction': critic_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_5']['answer'] + loop_results['stage_0.subtask_5']['thinking'],
            'temperature': 0.0,
            'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
        }
        results_0_2, log_0_2 = await self.reflexion(
            subtask_id="stage_0.subtask_2",
            reflect_desc=cot_reflect_desc_0_2,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_reflect_instruction_0_3 = (
            "Sub-task 3: Apply Kepler's third law and the orbital periods to estimate the total mass of each binary system, incorporating the mass ratio from subtask_2. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & former iterations of stage_0.subtask_5, respectively."
        )
        critic_instruction_0_3 = (
            "Please review and provide the limitations of provided solutions of total mass estimation from previous iterations."
        )
        cot_reflect_desc_0_3 = {
            'instruction': cot_reflect_instruction_0_3,
            'critic_instruction': critic_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_5']['answer'] + loop_results['stage_0.subtask_5']['thinking'],
            'temperature': 0.0,
            'context_desc': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
        }
        results_0_3, log_0_3 = await self.reflexion(
            subtask_id="stage_0.subtask_3",
            reflect_desc=cot_reflect_desc_0_3,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_reflect_instruction_0_4 = (
            "Sub-task 4: Compute the factor by which system_1 is more massive than system_2 by taking the ratio of their total masses. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3 & former iterations of stage_0.subtask_5, respectively."
        )
        critic_instruction_0_4 = (
            "Please review and provide the limitations of provided solutions of mass factor computation from previous iterations."
        )
        cot_reflect_desc_0_4 = {
            'instruction': cot_reflect_instruction_0_4,
            'critic_instruction': critic_instruction_0_4,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_5']['answer'] + loop_results['stage_0.subtask_5']['thinking'],
            'temperature': 0.0,
            'context_desc': ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
        }
        results_0_4, log_0_4 = await self.reflexion(
            subtask_id="stage_0.subtask_4",
            reflect_desc=cot_reflect_desc_0_4,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_reflect_instruction_0_5 = (
            "Sub-task 5: Refine and simplify the intermediate results to improve accuracy and clarity of the mass factor estimation. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4 & former iterations of stage_0.subtask_5, respectively."
        )
        critic_instruction_0_5 = (
            "Please review and provide the limitations of provided solutions of refinement from previous iterations."
        )
        cot_reflect_desc_0_5 = {
            'instruction': cot_reflect_instruction_0_5,
            'critic_instruction': critic_instruction_0_5,
            'input': [taskInfo] + loop_results['stage_0.subtask_4']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_5']['answer'] + loop_results['stage_0.subtask_5']['thinking'],
            'temperature': 0.0,
            'context_desc': ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
        }
        results_0_5, log_0_5 = await self.reflexion(
            subtask_id="stage_0.subtask_5",
            reflect_desc=cot_reflect_desc_0_5,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    debate_instruction_1_1 = (
        "Sub-task 1: Evaluate the refined mass factor estimates from stage_0 and select the candidate answer choice that best matches the computed factor. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the best candidate answer choice for the mass factor estimation problem."
    )
    debate_desc_1_1 = {
        'instruction': debate_instruction_1_1,
        'final_decision_instruction': final_decision_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
        'context_desc': ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"],
        'temperature': 0.5
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    aggregate_instruction_1_2 = (
        "Sub-task 2: Aggregate the evaluation results to finalize the best candidate answer choice. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    aggregate_desc_1_2 = {
        'instruction': aggregate_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.aggregate(
        subtask_id="stage_1.subtask_2",
        aggregate_desc=aggregate_desc_1_2
    )
    logs.append(log_1_2)

    review_instruction_2_1 = (
        "Sub-task 1: Review the selected candidate answer for correctness, consistency with physical laws, and alignment with the problem statement. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    review_desc_2_1 = {
        'instruction': review_instruction_2_1,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Perform a final step-by-step reasoning to confirm the validity of the selected answer and produce a final assessment. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    cot_agent_desc_2_2 = {
        'instruction': cot_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])
    return final_answer, logs
