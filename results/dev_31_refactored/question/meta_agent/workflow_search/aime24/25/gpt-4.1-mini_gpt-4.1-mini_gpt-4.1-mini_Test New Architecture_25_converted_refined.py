async def forward_25(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Summarize the given problem information, define variables for the hexagon and the triangle formed by extended lines, "
        "and clarify assumptions including vertex ordering and convexity. Input content are results (both thinking and answer) from: none."
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
        "Sub-task 2: Establish vector representations for the hexagon sides and express the parallelism and equilateral conditions as vector equations, "
        "explicitly defining vectors u, v, w corresponding to sides AB, CD, EF. Input content are results (both thinking and answer) from: stage_0.subtask_1."
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

    cot_instruction_0_3 = (
        "Sub-task 3: From the vector closure condition u + v + w = 0, rigorously derive the correct angle sum constraint for the directions of u, v, w, "
        "explicitly proving that the turning angles sum to 2π (not π), correcting the previous error where α + β + γ = π was incorrectly assumed. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Derive exact formulas relating the side lengths of the triangle formed by the extended lines (200, 240, 300) to the hexagon side length s and the angles between vectors u, v, w, "
        "using the corrected angle sum and vector geometry principles. Avoid assumptions invalidated in previous attempts. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_3."
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    cot_instruction_1_1 = (
        "Sub-task 1: Formulate a system of equations from the relations derived in stage_0.subtask_4 that link the hexagon side length s and the triangle side lengths, "
        "ensuring consistency with the vector closure and angle sum constraints. Input content are results (both thinking and answer) from: stage_0.subtask_4."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Solve the system of equations to find the numerical value of the hexagon side length s, carefully verifying each step to avoid propagation of previous errors. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Independently verify the derived angle relations and the candidate hexagon side length s by checking the vector closure condition u + v + w = 0 numerically or analytically, "
        "and confirm that the triangle side lengths and angle sums satisfy all geometric constraints, preventing the logical gaps from the previous attempt. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2 & stage_0.subtask_3 & stage_0.subtask_4."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer'], results_0_3['thinking'], results_0_3['answer'], results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
