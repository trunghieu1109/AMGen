async def forward_28(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and record the torus parameters: major radius R=6, minor radius r=3, "
        "and the sphere radius 11, ensuring clear notation for subsequent use. Input content: query provided."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Clarify the meaning of the two tangent configurations, define the tangent circle radii r_i and r_o on the torus, "
        "and specify the geometric interpretation of these radii in terms of the torus parameter theta, avoiding ambiguity. "
        "Input content: query and results from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Formulate the external tangency condition between the torus and sphere as an equation involving the torus parameter theta and the sphere center height h. "
        "Then algebraically eliminate h to derive a quadratic equation in cos(theta). This step must explicitly solve and simplify the equation rather than guessing or assuming forms, "
        "addressing the previous failure to solve the core tangency equation. Input content: results from stage_0.subtask_1 and stage_0.subtask_2."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Solve the quadratic equation in cos(theta) obtained from stage_1.subtask_1 to find the two distinct solutions cos(theta_i) and cos(theta_o) "
        "corresponding to the two tangent configurations. Verify the solutions are valid and consistent with the problem geometry, explicitly avoiding assumptions or guesses. "
        "Input content: results from stage_1.subtask_1."
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the quadratic solutions of cos(theta)."
    )
    cot_sc_desc_1_2 = {
        "instruction": cot_sc_instruction_1_2,
        "final_decision_instruction": final_decision_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc_1_2,
        n_repeat=self.max_sc
    )
    logs.append(log_1_2)

    cot_instruction_1_3 = (
        "Sub-task 3: Compute the tangent circle radii r_i and r_o on the torus using the formula r_i = R + r * cos(theta_i) and r_o = R + r * cos(theta_o), "
        "based on the solutions from stage_1.subtask_2. Confirm the correctness of these values before proceeding. "
        "Input content: results from stage_1.subtask_2 and stage_0.subtask_1."
    )
    cot_agent_desc_1_3 = {
        "instruction": cot_instruction_1_3,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer'], results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    cot_instruction_2_1 = (
        "Sub-task 1: Calculate the difference r_i - r_o using the values from stage_1.subtask_3, reduce the fraction to lowest terms m/n with m and n positive integers, "
        "and find the sum m + n. This step must be based strictly on the previously verified values to avoid guessing the final answer. "
        "Input content: results from stage_1.subtask_3."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
