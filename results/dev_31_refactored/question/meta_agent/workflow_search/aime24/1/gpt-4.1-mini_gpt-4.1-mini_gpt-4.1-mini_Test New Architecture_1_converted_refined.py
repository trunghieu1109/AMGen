async def forward_1(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and list all given elements, attributes, and constraints from the problem statement, "
        "ensuring clarity on points, segments, circle properties, and given side lengths. "
        "Input content are results (both thinking and answer) from: none."
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
        "Sub-task 2: Summarize relevant geometric properties and relationships (circle, tangents, chords, points) explicitly highlighting theorems and formulas that will be used later, "
        "such as power of a point and tangent-secant theorem, to prepare for symbolic reasoning and avoid numeric approximations. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
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
        "Sub-task 1: Using the given side lengths, symbolically determine key triangle parameters such as angles or segment lengths (e.g., using Law of Cosines or Law of Sines) necessary for subsequent exact computations, "
        "avoiding numeric approximations to prevent compounding errors. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Express the length of the tangent segments from point D to points B and C (i.e., DB = DC) symbolically in terms of known side lengths and triangle parameters, "
        "applying the power of a point theorem and tangent properties. This subtask explicitly addresses the previous failure of numeric slope approximations by focusing on exact algebraic expressions. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
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

    cot_instruction_1_3 = (
        "Sub-task 3: Set up and solve the tangent-secant length relation DB^2 = DA * DP symbolically, where DP = DA + AP, to derive a quadratic equation in AP. "
        "This step must avoid numeric substitution and instead use exact algebraic manipulation to find AP in terms of known lengths, directly addressing the previous error of numeric approximations and unjustified leaps. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2 & stage_1.subtask_1, respectively."
    )
    cot_agent_desc_1_3 = {
        "instruction": cot_instruction_1_3,
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    cot_instruction_2_1 = (
        "Sub-task 1: Solve the quadratic equation from stage_1.subtask_3 symbolically to find the exact value of AP as a reduced fraction m/n, ensuring m and n are coprime integers. "
        "Then compute and output the sum m + n. This subtask explicitly avoids numeric approximations and ensures the final answer is exact and justified. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_3, respectively."
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
