async def forward_2(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the coloring process and the rotational symmetry group of the octagon to understand how rotations act on vertex colorings, "
        "explicitly defining the group action and the event of interest. Input content are results (both thinking and answer) from: taskInfo."
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
        "Sub-task 2: Formally characterize the condition that there exists a rotation mapping all blue vertices to positions originally occupied by red vertices, "
        "clarifying the interpretation of 'original positions' and the fixed coloring before rotation. Input content are results (both thinking and answer) from: stage_0.subtask_1."
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
        "Sub-task 1: Validate the independence and equal probability coloring assumption and confirm the interpretation of rotations and vertex positions, "
        "ensuring no ambiguity remains about the problem setup and event definition. Input content are results (both thinking and answer) from: stage_0.subtask_2."
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

    cot_instruction_2_1 = (
        "Sub-task 1: Enumerate and verify the number of independent sets on small cycles C2 and C4 by explicit brute force to confirm the correct counts, "
        "addressing the previous error of misapplying the formula for independent sets on cycles. Input content are results (both thinking and answer) from: stage_1.subtask_1."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Generalize and verify the count of independent sets on the cycle C8, confirming it matches the Lucas number L8, "
        "and explicitly document the formula and reasoning to avoid previous miscounting. Input content are results (both thinking and answer) from: stage_2.subtask_1."
    )
    cot_agent_desc_2_2 = {
        "instruction": cot_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    cot_instruction_2_3 = (
        "Sub-task 3: For each rotation (by k vertices, k=0..7), determine the cycle decomposition of the rotation action on vertices and compute the number of colorings fixed by that rotation "
        "using the verified independent set counts on the corresponding cycles. Input content are results (both thinking and answer) from: stage_2.subtask_2."
    )
    cot_agent_desc_2_3 = {
        "instruction": cot_instruction_2_3,
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_2_3, log_2_3 = await self.cot(
        subtask_id="stage_2.subtask_3",
        cot_agent_desc=cot_agent_desc_2_3
    )
    logs.append(log_2_3)

    review_instruction_2_4 = (
        "Sub-task 4: Perform a consistency check by enumerating all 256 colorings and verifying the counts of colorings fixed by each rotation, "
        "ensuring the correctness of the combinatorial counts and formulas used. Input content are results (both thinking and answer) from: stage_2.subtask_3."
    )
    review_desc_2_4 = {
        "instruction": review_instruction_2_4,
        "input": [taskInfo, results_2_3['thinking'], results_2_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3"]
    }
    results_2_4, log_2_4 = await self.review(
        subtask_id="stage_2.subtask_4",
        review_desc=review_desc_2_4
    )
    logs.append(log_2_4)

    cot_instruction_3_1 = (
        "Sub-task 1: Apply the inclusion–exclusion principle to combine the counts of colorings fixed by each rotation, "
        "carefully computing the size of the union of these sets to avoid the previous omission of overlap correction. Input content are results (both thinking and answer) from: stage_2.subtask_3 & stage_2.subtask_4."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo, results_2_3['thinking'], results_2_3['answer'], results_2_4['thinking'], results_2_4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3", "thinking of stage_2.subtask_4", "answer of stage_2.subtask_4"]
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    cot_sc_instruction_3_2 = (
        "Sub-task 2: Calculate the final probability as a reduced fraction m/n from the inclusion–exclusion result and compute m+n, "
        "ensuring the fraction is in lowest terms. Input content are results (both thinking and answer) from: stage_3.subtask_1."
    )
    final_decision_instruction_3_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the final probability calculation and sum m+n."
    )
    cot_sc_desc_3_2 = {
        "instruction": cot_sc_instruction_3_2,
        "final_decision_instruction": final_decision_instruction_3_2,
        "input": [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_3_2, log_3_2 = await self.sc_cot(
        subtask_id="stage_3.subtask_2",
        cot_agent_desc=cot_sc_desc_3_2,
        n_repeat=self.max_sc
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
