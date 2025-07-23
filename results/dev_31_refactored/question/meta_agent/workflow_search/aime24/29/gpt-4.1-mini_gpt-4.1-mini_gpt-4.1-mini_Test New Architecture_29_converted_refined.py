async def forward_29(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Define the grid model, chip color groups, and formalize the constraints on row and column color uniformity. "
        "Clearly specify that each cell can hold at most one chip and chips are indistinguishable within colors. "
        "Input content are results (both thinking and answer) from: taskInfo."
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
        "Sub-task 2: Explicitly formalize the maximality condition at the cell level: for each empty cell (i,j), define the precise condition that placing a white or black chip there would violate the uniformity constraints. "
        "Derive the equivalent global constraints on row and column color assignments from this per-cell maximality condition. "
        "This addresses the previous failure of mischaracterizing maximality at the row/column level. "
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

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Analyze how row and column color assignments interact under the refined per-cell maximality condition to produce valid maximal chip placements. "
        "Classify possible maximal configurations by their row and column color patterns, ensuring the blocking conditions at each empty cell are satisfied. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_0.subtask_2, respectively."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for classification of maximal configurations."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Validate the characterization of maximal configurations by implementing a brute-force enumeration on smaller grids (2x2 and 3x3). "
        "Confirm that the per-cell maximality condition and classification from stage_1.subtask_1 correctly capture all maximal configurations. "
        "This step prevents propagation of earlier errors and builds confidence before scaling to 5x5. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    critic_instruction_1_2 = (
        "Please review and provide the limitations of provided solutions of stage_1.subtask_1 and confirm correctness of maximal configurations classification."
    )
    cot_reflect_desc_1_2 = {
        "instruction": cot_reflect_instruction_1_2,
        "critic_instruction": critic_instruction_1_2,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Apply combinatorial calculations to count the number of maximal configurations on the 5x5 grid based on the validated classification and per-cell maximality conditions. "
        "Use insights from the small-grid validation to ensure correctness of the counting method. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_1.subtask_2, respectively."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the count of maximal configurations on 5x5 grid."
    )
    cot_sc_desc_2_1 = {
        "instruction": cot_sc_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc_2_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate the computed counts, verify consistency with all problem constraints including maximality, and finalize the total number of valid maximal placements. "
        "Cross-check with earlier formalizations and small-grid results to ensure no contradictions remain. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_0.subtask_2 & stage_1.subtask_1 & stage_1.subtask_2 & stage_2.subtask_1, respectively."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer'], results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer'], results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
