async def forward_2(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the coloring process and the rotational symmetry group of the octagon to understand how rotations act on vertex colorings. "
        "Input content: taskInfo (query and detailed analysis)."
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
        "Sub-task 2: Characterize the condition that a rotation maps all blue vertices to positions originally occupied by red vertices. "
        "Input content: results (thinking and answer) from stage_0.subtask_1."
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

    review_instruction_1_1 = (
        "Sub-task 1: Validate the independence and equal probability coloring assumption and confirm the interpretation of rotations and vertex positions. "
        "Input content: results (thinking and answer) from stage_0.subtask_2."
    )
    review_desc_1_1 = {
        "instruction": review_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.review(
        subtask_id="stage_1.subtask_1",
        review_desc=review_desc_1_1
    )
    logs.append(log_1_1)

    review_instruction_2_1 = (
        "Sub-task 1: Identify and enumerate the rotations and colorings that satisfy the condition of blue vertices mapping to original red vertices. "
        "Input content: results (thinking and answer) from stage_1.subtask_1."
    )
    review_desc_2_1 = {
        "instruction": review_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Calculate the probability as a reduced fraction m/n and compute m+n. "
        "Input content: results (thinking and answer) from stage_2.subtask_1 and stage_1.subtask_1."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the probability calculation and final sum m+n."
    )
    cot_sc_desc_3_1 = {
        "instruction": cot_sc_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_3_1, log_3_1 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc_3_1,
        n_repeat=self.max_sc
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
