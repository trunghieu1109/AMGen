async def forward_170(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize chemical information about each substituent and its directing effects on electrophilic substitution. "
        "Input: taskInfo containing the list of substances and their substituents."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Characterize the expected regioselectivity (ortho/para/meta directing) and steric factors influencing para-isomer yield for each substance. "
        "Input: taskInfo and results (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_reflect_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Generate an initial ranking of substances by increasing para-isomer weight fraction based on substituent effects. "
            "Input: taskInfo and all previous results (thinking and answer) from stage_0.subtask_2 and previous iterations of stage_1.subtask_2 if any."
        )
        cot_reflect_desc_1_1 = {
            "instruction": cot_reflect_instruction_1_1,
            "critic_instruction": "Please review and provide the limitations of the initial ranking and suggest improvements.",
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage_1['stage_1.subtask_2']['answer'] + loop_results_stage_1['stage_1.subtask_2']['thinking'],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["answer of previous iteration stage_1.subtask_2"]*len(loop_results_stage_1['stage_1.subtask_2']['answer']) + ["thinking of previous iteration stage_1.subtask_2"]*len(loop_results_stage_1['stage_1.subtask_2']['thinking'])
        }
        results_1_1, log_1_1 = await self.reflexion(
            subtask_id="stage_1.subtask_1",
            reflect_desc=cot_reflect_desc_1_1,
            n_repeat=1
        )
        logs.append(log_1_1)

        loop_results_stage_1['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results_stage_1['stage_1.subtask_1']['answer'].append(results_1_1['answer'])

        cot_reflect_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine the ranking by considering steric hindrance and possible exceptions, using outputs from previous iteration. "
            "Input: taskInfo and all previous results (thinking and answer) from stage_1.subtask_1 and previous iterations of stage_1.subtask_2."
        )
        cot_reflect_desc_1_2 = {
            "instruction": cot_reflect_instruction_1_2,
            "critic_instruction": "Please review and provide the limitations of the refined ranking and suggest further improvements.",
            "input": [taskInfo] + loop_results_stage_1['stage_1.subtask_1']['answer'] + loop_results_stage_1['stage_1.subtask_1']['thinking'] + loop_results_stage_1['stage_1.subtask_2']['answer'] + loop_results_stage_1['stage_1.subtask_2']['thinking'],
            "temperature": 0.0,
            "context_desc": ["user query"] + ["answer of stage_1.subtask_1"]*len(loop_results_stage_1['stage_1.subtask_1']['answer']) + ["thinking of stage_1.subtask_1"]*len(loop_results_stage_1['stage_1.subtask_1']['thinking']) + ["answer of previous iteration stage_1.subtask_2"]*len(loop_results_stage_1['stage_1.subtask_2']['answer']) + ["thinking of previous iteration stage_1.subtask_2"]*len(loop_results_stage_1['stage_1.subtask_2']['thinking'])
        }
        results_1_2, log_1_2 = await self.reflexion(
            subtask_id="stage_1.subtask_2",
            reflect_desc=cot_reflect_desc_1_2,
            n_repeat=1
        )
        logs.append(log_1_2)

        loop_results_stage_1['stage_1.subtask_2']['thinking'].append(results_1_2['thinking'])
        loop_results_stage_1['stage_1.subtask_2']['answer'].append(results_1_2['answer'])

    debate_instruction_2_1 = (
        "Sub-task 1: Compare the refined ranking with the provided answer choices and select the best matching order. "
        "Input: taskInfo and all results (thinking and answer) from stage_1.subtask_2 iterations."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": "Sub-task 1: Select the best matching order of substances by increasing para-isomer weight fraction from the given choices.",
        "input": [taskInfo] + loop_results_stage_1['stage_1.subtask_2']['thinking'] + loop_results_stage_1['stage_1.subtask_2']['answer'],
        "context_desc": ["user query"] + ["thinking of stage_1.subtask_2"]*len(loop_results_stage_1['stage_1.subtask_2']['thinking']) + ["answer of stage_1.subtask_2"]*len(loop_results_stage_1['stage_1.subtask_2']['answer']),
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_2_1,
        n_repeat=1
    )
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Sub-task 1: Assess the selected ranking for consistency, correctness, and alignment with chemical principles, producing final feedback. "
        "Input: taskInfo and results (thinking and answer) from stage_2.subtask_1."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_2_1['answer'])
    return final_answer, logs
