async def forward_27(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Formulate the problem constraints and represent the condition that changing any digit of N to 1 yields a number divisible by 7. "
        "Input content are results (both thinking and answer) from: taskInfo."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    # stage_0.subtask_2
    cot_instruction_0_2 = (
        "Stage 0, Sub-task 2: Generate a systematic approach to test four-digit numbers from 9999 downwards against the divisibility conditions. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    # stage_1.subtask_1
    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Apply the approach to identify the greatest four-digit number N satisfying the divisibility property. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    # stage_1.subtask_2
    cot_instruction_1_2 = (
        "Stage 1, Sub-task 2: Compute Q and R where Q is the quotient and R the remainder when N is divided by 1000. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    loop_results = {
        "stage_2.subtask_1": {"thinking": [], "answer": []},
        "stage_3.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(3):
        # stage_2.subtask_1
        cot_reflect_instruction_2_1 = (
            "Stage 2, Sub-task 1: For each digit position in N, replace the digit with 1 and check divisibility by 7. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1 & former iterations of stage_3.subtask_1, respectively."
        )
        critic_instruction_2_1 = (
            "Stage 2, Sub-task 1, Criticism: Review and provide limitations of divisibility checks from stage_1.subtask_1 and previous stage_3.subtask_1 iterations."
        )
        inputs_2_1 = [taskInfo, results_1_1['thinking'], results_1_1['answer']]
        if loop_results["stage_3.subtask_1"]["thinking"]:
            inputs_2_1 += loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"]

        cot_reflect_desc_2_1 = {
            "instruction": cot_reflect_instruction_2_1,
            "critic_instruction": critic_instruction_2_1,
            "input": inputs_2_1,
            "temperature": 0.7,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"] +
                            (["thinking of previous stage_3.subtask_1 iterations", "answer of previous stage_3.subtask_1 iterations"] if loop_results["stage_3.subtask_1"]["thinking"] else [])
        }
        results_2_1, log_2_1 = await self.reflexion(subtask_id="stage_2.subtask_1", reflect_desc=cot_reflect_desc_2_1, n_repeat=1)
        logs.append(log_2_1)

        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

        # stage_3.subtask_1
        debate_instruction_3_1 = (
            "Stage 3, Sub-task 1: Assess the correctness of divisibility checks and provide feedback for refinement if needed. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_2.subtask_1, respectively."
        )
        final_decision_instruction_3_1 = (
            "Stage 3, Sub-task 1, Final Decision: Evaluate and validate divisibility checks correctness and provide refined answer."
        )
        inputs_3_1 = [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_2_1['thinking'], results_2_1['answer']]

        debate_desc_3_1 = {
            "instruction": debate_instruction_3_1,
            "final_decision_instruction": final_decision_instruction_3_1,
            "input": inputs_3_1,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
            "temperature": 0.5
        }
        results_3_1, log_3_1 = await self.sc_cot(subtask_id="stage_3.subtask_1", cot_agent_desc=debate_desc_3_1, n_repeat=3)
        logs.append(log_3_1)

        loop_results["stage_3.subtask_1"]["thinking"].append(results_3_1["thinking"])
        loop_results["stage_3.subtask_1"]["answer"].append(results_3_1["answer"])

    final_answer = await self.make_final_answer(loop_results["stage_3.subtask_1"]['thinking'][-1], loop_results["stage_3.subtask_1"]['answer'][-1])
    return final_answer, logs
