async def forward_9(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Calculate total number of ways Jen can pick 4 distinct numbers and total ways 4 numbers can be randomly chosen from S. "
        "Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    # stage_0.subtask_2
    cot_reflect_instruction_0_2 = (
        "Stage 0, Sub-task 2: Compute counts of outcomes where Jen wins a prize (at least 2 matches) and where Jen wins the grand prize (all 4 match). "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    critic_instruction_0_2 = (
        "Stage 0, Sub-task 2, Criticism: Please review and provide the limitations of provided solutions of stage_0.subtask_2."
    )
    cot_reflect_desc_0_2 = {
        'instruction': cot_reflect_instruction_0_2,
        'critic_instruction': critic_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.reflexion(subtask_id='stage_0.subtask_2', reflect_desc=cot_reflect_desc_0_2, n_repeat=2)
    logs.append(log_0_2)

    # stage_1.subtask_1
    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Determine the number of favorable outcomes for winning a prize and for winning the grand prize from the counts derived. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    # stage_2.subtask_1
    cot_instruction_2_1 = (
        "Stage 2, Sub-task 1: Express the conditional probability of winning the grand prize given winning a prize as a reduced fraction m/n and compute m+n. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    loop_results_stage_3 = {
        'stage_3.subtask_1': {'thinking': [], 'answer': []}
    }

    for i in range(3):
        formatter_instruction_3_1 = (
            "Stage 3, Sub-task 1: Iteratively generate and refine intermediate combinatorial calculations and probability simplifications to improve accuracy and clarity. "
            "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
        )
        formatter_desc_3_1 = {
            'instruction': formatter_instruction_3_1,
            'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'],
            'format': 'short and concise, without explaination'
        }
        results_3_1, log_3_1 = await self.specific_format(subtask_id='stage_3.subtask_1', formatter_desc=formatter_desc_3_1)
        loop_results_stage_3['stage_3.subtask_1']['thinking'].append(results_3_1['thinking'])
        loop_results_stage_3['stage_3.subtask_1']['answer'].append(results_3_1['answer'])
        logs.append(log_3_1)

    debate_instruction_6_1 = (
        "Stage 6, Sub-task 1: Consolidate and finalize the simplified fraction and sum m+n into a clear, concise final answer. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1 & former iterations of stage_3.subtask_1, respectively."
    )
    final_decision_instruction_6_1 = (
        "Stage 6, Sub-task 1, Final Decision: Consolidate and finalize the simplified fraction and sum m+n into a clear, concise final answer."
    )
    debate_desc_6_1 = {
        'instruction': debate_instruction_6_1,
        'final_decision_instruction': final_decision_instruction_6_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']] + loop_results_stage_3['stage_3.subtask_1']['thinking'] + loop_results_stage_3['stage_3.subtask_1']['answer'],
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'] + ['thinking of stage_3.subtask_1 iteration']*3 + ['answer of stage_3.subtask_1 iteration']*3,
        'temperature': 0.5
    }
    results_6_1, log_6_1 = await self.debate(subtask_id='stage_6.subtask_1', debate_desc=debate_desc_6_1, n_repeat=2)
    logs.append(log_6_1)

    final_answer = await self.make_final_answer(results_6_1['thinking'], results_6_1['answer'])
    return final_answer, logs
