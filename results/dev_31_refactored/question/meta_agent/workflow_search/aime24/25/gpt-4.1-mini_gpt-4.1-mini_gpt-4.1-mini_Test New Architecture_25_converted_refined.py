async def forward_25(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Derive the relationship among the three directions of lines AB, CD, and EF imposed by the hexagon being equilateral with opposite sides parallel, specifically proving that the sum of the three external angles between these lines is 2π. This subtask must avoid conflating vector sums with line intersections and explicitly justify the angular constraints from the hexagon's geometry. Input content are results (both thinking and answer) from: none."
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

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Stage 1, Sub-task 1: Using the angular relationships from stage_0.subtask_1, correctly express each side length of the triangle formed by the extensions of AB, CD, and EF in terms of the hexagon side length s and the three external angles. Apply the law of sines and/or cosines appropriately to the intersections of the lines, avoiding the incorrect assumption that triangle sides equal vector sums of hexagon sides. Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively. Iteration {iteration+1} of 2."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1['thinking'])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1['answer'])

        cot_sc_instruction_2_1 = (
            f"Stage 2, Sub-task 1: Formulate and solve the system of equations derived from the expressions of the triangle side lengths (200, 240, 300) in terms of s and the angles from stage_1.subtask_1. Validate the solution for s by checking consistency with all geometric constraints and the given triangle side lengths. This subtask must explicitly avoid assuming classical results without proof and instead rely on derived formulas. Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1, respectively. Iteration {iteration+1} of 2."
        )
        final_decision_instruction_2_1 = (
            "Stage 2, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for solving the system of equations for s."
        )
        cot_sc_desc_2_1 = {
            "instruction": cot_sc_instruction_2_1,
            "final_decision_instruction": final_decision_instruction_2_1,
            "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_1_1['thinking'], results_1_1['answer']],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_2_1, log_2_1 = await self.sc_cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_sc_desc_2_1,
            n_repeat=self.max_sc
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1['thinking'])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1['answer'])

    cot_instruction_2_2 = (
        "Stage 2, Sub-task 2: Derive the final numeric value of the hexagon side length s by applying geometric constraints and algebraic simplifications to the validated solution from stage_2.subtask_1. Ensure the final answer is consistent with all prior deductions and clearly justified. Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1 & stage_2.subtask_1, respectively."
    )
    cot_agent_desc_2_2 = {
        "instruction": cot_instruction_2_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], loop_results["stage_1.subtask_1"]["thinking"], loop_results["stage_1.subtask_1"]["answer"], loop_results["stage_2.subtask_1"]["thinking"], loop_results["stage_2.subtask_1"]["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])
    return final_answer, logs
