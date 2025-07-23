async def forward_150(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}}

    for i in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and normalize the given state vector and observable matrix operator P. "
            "Input content: taskInfo"
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
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Compute the eigenvalues and eigenvectors of the operator P. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 iterations"
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
        logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Identify the eigenspace corresponding to eigenvalue 0 and construct the projection operator onto this eigenspace. "
        "Input content: taskInfo, all thinking and answers from stage_0.subtask_2 iterations"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        "Sub-task 1: Project the normalized state vector onto the zero eigenspace and calculate the probability of measuring eigenvalue 0. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_3_1 = (
        "Sub-task 1: Validate the calculated probability for correctness and consistency with quantum mechanical principles. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1 and stage_2.subtask_1"
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
