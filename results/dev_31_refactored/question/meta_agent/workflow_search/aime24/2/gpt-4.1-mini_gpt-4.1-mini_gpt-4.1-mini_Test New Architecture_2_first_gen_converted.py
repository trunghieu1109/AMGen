async def forward_2(self, taskInfo):
    logs = []

    # Stage 0, Sub-task 1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Analyze the problem setup: define the coloring scheme, the symmetry group of the octagon, "
        "and the condition for rotation mapping blue vertices to originally red vertices. Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    # Stage 0, Sub-task 2
    cot_sc_instruction_0_2 = (
        "Stage 0, Sub-task 2: Construct intermediate mathematical representations such as the group action on vertex colorings "
        "and characterize the event of interest in terms of these representations. Input content: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    final_decision_instruction_0_2 = (
        "Stage 0, Sub-task 2, Final Decision: Synthesize and choose the most consistent answer for constructing mathematical representations."
    )
    cot_sc_desc_0_2 = {
        "instruction": cot_sc_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_sc_desc_0_2, n_repeat=self.max_sc)
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Stage 1, Sub-task 1: Combine the intermediate representations with probability calculations for each rotation "
            f"to estimate the probability that a rotation maps blue vertices to originally red vertices. "
            f"Input content: taskInfo, thinking and answer from stage_0.subtask_2, and all previous thinking and answers from stage_2.subtask_1 iterations. Iteration {iteration+1}"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage_1["stage_2.subtask_1"]["thinking"] + loop_results_stage_1["stage_2.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1['thinking'])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1['answer'])

        cot_sc_instruction_2_1 = (
            f"Stage 2, Sub-task 1: Evaluate and validate the consolidated probabilities for each rotation, "
            f"selecting those rotations that satisfy the problem's condition and refining the probability estimate. "
            f"Input content: thinking and answer from stage_1.subtask_1 iteration {iteration+1}"
        )
        final_decision_instruction_2_1 = (
            "Stage 2, Sub-task 1, Final Decision: Synthesize and choose the most consistent refined probability estimate."
        )
        cot_sc_desc_2_1 = {
            "instruction": cot_sc_instruction_2_1,
            "final_decision_instruction": final_decision_instruction_2_1,
            "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_2_1, log_2_1 = await self.sc_cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_sc_desc_2_1, n_repeat=self.max_sc)
        logs.append(log_2_1)
        loop_results_stage_1["stage_2.subtask_1"]["thinking"].append(results_2_1['thinking'])
        loop_results_stage_1["stage_2.subtask_1"]["answer"].append(results_2_1['answer'])

    cot_agent_instruction_3_1 = (
        "Stage 3, Sub-task 1: Derive the final simplified probability fraction m/n and compute m+n as the final answer. "
        "Input content: thinking and answer from stage_2.subtask_1 from all iterations."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo] + loop_results_stage_1["stage_2.subtask_1"]["thinking"] + loop_results_stage_1["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_3_1, log_3_1 = await self.answer_generate(subtask_id="stage_3.subtask_1", cot_agent_desc=cot_agent_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
