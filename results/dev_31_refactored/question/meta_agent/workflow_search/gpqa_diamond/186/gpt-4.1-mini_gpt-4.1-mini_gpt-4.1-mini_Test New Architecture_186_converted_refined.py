async def forward_186(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract star data from the query and calculate apparent magnitudes for all listed stars using their absolute magnitudes and distances, "
        "ensuring accurate brightness values for subsequent detectability analysis. Input: [taskInfo]"
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
        "stage_1.subtask_2": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            "Sub-task 1: Retrieve the official ESPRESSO spectrograph performance data (e.g., S/N vs. V-magnitude curves or tables) "
            "from the provided ESO overview link or trusted documentation, ensuring no assumptions are made about sensitivity limits. "
            "Input: [taskInfo]"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])
        logs.append(log_1_1)

        cot_sc_instruction_1_2 = (
            "Sub-task 2: Analyze and interpolate the retrieved ESPRESSO performance data to determine the minimum apparent V magnitude at which S/N ≥ 10 per binned pixel is achievable in a 1-hour exposure with an 8m VLT telescope. "
            "This threshold must be firmly grounded in documented instrument specs to avoid prior errors. "
            "Input: [taskInfo, all previous thinking and answers from stage_1.subtask_1 iterations]"
        )
        final_decision_instruction_1_2 = (
            "Sub-task 2: Synthesize and choose the most consistent sensitivity threshold for ESPRESSO detectability based on all retrieved data."
        )
        cot_sc_desc_1_2 = {
            "instruction": cot_sc_instruction_1_2,
            "final_decision_instruction": final_decision_instruction_1_2,
            "input": [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.5,
            "context": ["user query"] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"]
        }
        results_1_2, log_1_2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_1_2,
            n_repeat=self.max_sc
        )
        loop_results["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])
        logs.append(log_1_2)

        cot_sc_instruction_2_1 = (
            "Sub-task 1: Compare each star's apparent magnitude (from stage_0.subtask_1) to the interpolated ESPRESSO sensitivity threshold (from stage_1.subtask_2) to classify stars as detectable or not. "
            "This subtask explicitly incorporates the corrected sensitivity threshold to avoid misclassification from previous attempts. "
            "Input: [taskInfo, all previous thinking and answers from stage_0.subtask_1 and stage_1.subtask_2 iterations]"
        )
        final_decision_instruction_2_1 = (
            "Sub-task 1: Synthesize and choose the most consistent detectability classification for all stars based on apparent magnitudes and sensitivity threshold."
        )
        cot_sc_desc_2_1 = {
            "instruction": cot_sc_instruction_2_1,
            "final_decision_instruction": final_decision_instruction_2_1,
            "input": [taskInfo] + [results_0_1["thinking"], results_0_1["answer"]] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"],
            "temperature": 0.5,
            "context": ["user query", results_0_1["thinking"], results_0_1["answer"]] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"]
        }
        results_2_1, log_2_1 = await self.sc_cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_sc_desc_2_1,
            n_repeat=self.max_sc
        )
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1["thinking"])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1["answer"])
        logs.append(log_2_1)

    cot_agent_instruction_3_1 = (
        "Sub-task 1: Summarize the total number of detectable stars based on classification results from stage_2.subtask_1 and format the final answer according to the provided multiple-choice options. "
        "Input: [taskInfo, all thinking and answers from stage_2.subtask_1 iterations]"
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
        "temperature": 0.0,
        "context": ["user query"] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"]
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
