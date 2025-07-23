async def forward_190(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}}
    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Analyze and document the chemical transformations from starting material through product 1 and product 2, "
            "including functional group changes and intermediate structures. Input: taskInfo only."
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
            "Sub-task 2: Analyze and document the transformations from product 2 through product 3 and product 4, "
            "including mechanistic implications and structural changes. Input: taskInfo, all previous thinking and answers from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query"] + ["thinking of stage_0.subtask_1"]*len(loop_results["stage_0.subtask_1"]["thinking"]) + ["answer of stage_0.subtask_1"]*len(loop_results["stage_0.subtask_1"]["answer"])
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
        logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Integrate the intermediate structures and transformations from stage_0 to deduce the final product 4 structure "
        "and rationalize the reaction sequence. Input: taskInfo, all thinking and answers from all iterations of stage_0.subtask_1 and stage_0.subtask_2."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query"] + ["thinking of stage_0.subtask_1"]*len(loop_results["stage_0.subtask_1"]["thinking"]) + ["answer of stage_0.subtask_1"]*len(loop_results["stage_0.subtask_1"]["answer"]) + ["thinking of stage_0.subtask_2"]*len(loop_results["stage_0.subtask_2"]["thinking"]) + ["answer of stage_0.subtask_2"]*len(loop_results["stage_0.subtask_2"]["answer"])
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_agent_instruction_2_1 = (
        "Sub-task 1: Evaluate the candidate product structures against the deduced final product 4 structure "
        "and select the best matching candidate. Input: taskInfo, thinking and answer from stage_1.subtask_1."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_agent_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.answer_generate(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1["thinking"], results_2_1["answer"])
    return final_answer, logs
