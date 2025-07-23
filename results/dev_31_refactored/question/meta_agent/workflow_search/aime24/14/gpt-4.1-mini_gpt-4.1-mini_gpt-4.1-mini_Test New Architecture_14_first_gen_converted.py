async def forward_14(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Express coordinates of points A, B, C, D on the hyperbola with diagonals intersecting at the origin and satisfying rhombus properties. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
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

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, derive algebraic relationships representing equal side lengths and perpendicular diagonals of the rhombus. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for deriving algebraic relationships representing equal side lengths and perpendicular diagonals of the rhombus."
    )
    cot_sc_desc_0_2 = {
        "instruction": cot_sc_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Calculate formulas for the squares of the diagonals AC and BD in terms of parameters from stage_0. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Determine constraints on parameters to ensure points lie on the hyperbola and form a rhombus, then find the supremum of BD^2 under these constraints. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_1, respectively."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for determining constraints and supremum of BD^2."
    )
    cot_sc_desc_2_1 = {
        "instruction": cot_sc_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc_2_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    debate_instruction_3_1 = (
        "Sub-task 1: Simplify and consolidate the supremum expression to produce a clear final answer. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Refine and finalize the greatest real number less than BD^2 for all such rhombi."
    )
    debate_desc_3_1 = {
        "instruction": debate_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_3_1, log_3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
