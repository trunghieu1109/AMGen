async def forward_2(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Define the problem setup: represent the octagon vertices, coloring scheme, and the rotational symmetry group. "
        "Input content: [taskInfo]"
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

    # stage_0.subtask_2
    cot_instruction_0_2 = (
        "Stage 0, Sub-task 2: Formulate the condition for a rotation to map blue vertices onto originally red vertices and express it in terms of sets and group actions. "
        "Input content: results (thinking and answer) from stage_0.subtask_1"
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

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for i in range(3):
        cot_instruction_1_1 = (
            f"Stage 1, Sub-task 1 (iteration {i+1}): For each rotation, determine the number of colorings where blue vertices map to red vertices under that rotation. "
            "Input content: results (thinking and answer) from stage_0.subtask_2 and all former iterations of stage_2.subtask_1"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["thinking of previous stage_2.subtask_1 iterations"]*len(loop_results["stage_2.subtask_1"]["thinking"])
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_2_1 = (
            f"Stage 2, Sub-task 1 (iteration {i+1}): Calculate the probability that at least one rotation satisfies the blue-to-red mapping condition by applying inclusion-exclusion or Burnside's lemma. "
            "Input content: results (thinking and answer) from stage_1.subtask_1"
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

    cot_agent_instruction_3_1 = (
        "Stage 3, Sub-task 1: Simplify the probability fraction to lowest terms and compute the sum m+n for the final answer. "
        "Input content: results (thinking and answer) from stage_2.subtask_1"
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
