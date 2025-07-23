async def forward_196(self, taskInfo):
    logs = []
    loop_results = {"stage_2.subtask_1": {"thinking": [], "answer": []}, "stage_3.subtask_1": {"thinking": [], "answer": []}}

    cot_instruction_0_1 = (
        "Sub-task 1: Interpret IR and 1H NMR data to identify functional groups and substitution patterns in Compound X. "
        "Input: taskInfo containing question and spectral data."
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
        "Sub-task 2: Analyze the reaction conditions with red phosphorus and HI to predict possible chemical transformations. "
        "Input: taskInfo containing question and reaction conditions."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Sub-task 1: Combine spectral interpretation and reaction analysis to deduce possible structures of the final product. "
        "Input: results from stage_0.subtask_1 and stage_0.subtask_2 (thinking and answer)."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"], results_0_2["thinking"], results_0_2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    for iteration in range(2):
        debate_instruction_2_1 = (
            "Sub-task 1: Evaluate each candidate product against the deduced structural features and reaction outcome to score their likelihood. "
            "Input: results from stage_1.subtask_1 and all previous outputs of stage_3.subtask_1 (thinking and answer) from prior iterations."
        )
        debate_desc_2_1 = {
            "instruction": debate_instruction_2_1,
            "final_decision_instruction": "Sub-task 1: Select the best candidate product based on evaluation.",
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"],
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"] + ["thinking of stage_3.subtask_1 iteration {}".format(i+1) for i in range(len(loop_results["stage_3.subtask_1"]["thinking"]))] + ["answer of stage_3.subtask_1 iteration {}".format(i+1) for i in range(len(loop_results["stage_3.subtask_1"]["answer"]))],
            "temperature": 0.5
        }
        results_2_1, log_2_1 = await self.debate(
            subtask_id="stage_2.subtask_1",
            debate_desc=debate_desc_2_1,
            n_repeat=1
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])

        debate_instruction_3_1 = (
            "Sub-task 1: Critically assess the selected candidate for chemical validity and consistency with all data, refining the choice if needed. "
            "Input: results from stage_2.subtask_1 (thinking and answer) of all iterations."
        )
        debate_desc_3_1 = {
            "instruction": debate_instruction_3_1,
            "final_decision_instruction": "Sub-task 1: Provide a refined and validated candidate product.",
            "input": [taskInfo] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
            "context_desc": ["user query"] + ["thinking of stage_2.subtask_1 iteration {}".format(i+1) for i in range(len(loop_results["stage_2.subtask_1"]["thinking"]))] + ["answer of stage_2.subtask_1 iteration {}".format(i+1) for i in range(len(loop_results["stage_2.subtask_1"]["answer"]))],
            "temperature": 0.5
        }
        results_3_1, log_3_1 = await self.debate(
            subtask_id="stage_3.subtask_1",
            debate_desc=debate_desc_3_1,
            n_repeat=1
        )
        logs.append(log_3_1)
        loop_results["stage_3.subtask_1"]["thinking"].append(results_3_1["thinking"])
        loop_results["stage_3.subtask_1"]["answer"].append(results_3_1["answer"])

    cot_agent_instruction_4_1 = (
        "Sub-task 1: Consolidate the validated candidate into a final answer identifying the product. "
        "Input: results from stage_3.subtask_1 (thinking and answer) of all iterations."
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_agent_instruction_4_1,
        "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query"] + ["thinking of stage_3.subtask_1 iteration {}".format(i+1) for i in range(len(loop_results["stage_3.subtask_1"]["thinking"]))] + ["answer of stage_3.subtask_1 iteration {}".format(i+1) for i in range(len(loop_results["stage_3.subtask_1"]["answer"]))]
    }
    results_4_1, log_4_1 = await self.answer_generate(
        subtask_id="stage_4.subtask_1",
        cot_agent_desc=cot_agent_desc_4_1
    )
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
