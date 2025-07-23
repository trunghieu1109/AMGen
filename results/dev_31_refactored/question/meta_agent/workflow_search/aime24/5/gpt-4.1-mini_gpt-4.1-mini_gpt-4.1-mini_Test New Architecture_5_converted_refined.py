async def forward_5(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Determine a consistent coordinate representation or vector model of tetrahedron ABCD that exactly satisfies the given edge length equalities and symmetries, "
        "avoiding numeric approximations to prevent propagation of errors. Input content are results (both thinking and answer) from: taskInfo."
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

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Compute the exact areas of the four faces of the tetrahedron using the coordinate or vector representation from stage_0.subtask_1, "
        "ensuring symbolic radical expressions are preserved to avoid approximation errors. Input content are results (both thinking and answer) from: stage_0.subtask_1."
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for exact face areas of tetrahedron ABCD."
    )
    cot_sc_desc_0_2 = {
        "instruction": cot_sc_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Compute the volume of tetrahedron ABCD symbolically using the Cayley-Menger determinant or an equivalent exact formula, "
        "carrying radicals exactly to avoid conflicting numeric approximations. Input content are results (both thinking and answer) from: stage_0.subtask_1."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the volume of tetrahedron ABCD."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Verify the computed volume from stage_1.subtask_1 by an independent method (e.g., triple scalar product from coordinates or alternative formula) to ensure consistency and correctness before using it in further calculations. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_0.subtask_1."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Use the verified volume from stage_1.subtask_2 and the total face area from stage_0.subtask_2 to compute the inradius of the tetrahedron via the formula r = 3V / (sum of face areas), "
        "ensuring exact symbolic manipulation without premature numeric approximation. Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_2."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"], results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Simplify the inradius expression obtained in stage_2.subtask_1 into the form (m√n)/p, where m, n, p are positive integers, m and p are coprime, and n is square-free. "
        "This subtask explicitly incorporates reflection and critique to avoid errors in simplification and ensure the final answer meets problem requirements. Input content are results (both thinking and answer) from: stage_2.subtask_1."
    )
    critic_instruction_2_2 = (
        "Please review and provide the limitations of provided solutions of simplification of the inradius expression, ensuring the final form is correct and meets problem requirements."
    )
    cot_reflect_desc_2_2 = {
        "instruction": cot_reflect_instruction_2_2,
        "critic_instruction": critic_instruction_2_2,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.reflexion(
        subtask_id="stage_2.subtask_2",
        reflect_desc=cot_reflect_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2["thinking"], results_2_2["answer"])
    return final_answer, logs
