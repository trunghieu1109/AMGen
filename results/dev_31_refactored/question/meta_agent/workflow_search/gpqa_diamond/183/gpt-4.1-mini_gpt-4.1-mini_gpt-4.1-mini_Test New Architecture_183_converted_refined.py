async def forward_183(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    results_stage_0 = {}

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze each provided reaction sequence stepwise to identify intermediate transformations, their chemical rationale, and expected regioselectivity, explicitly checking directing effects of substituents at each electrophilic aromatic substitution (EAS) step. Input: taskInfo"
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
    results_stage_0["stage_0.subtask_1"] = results_0_1

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Validate the regioselectivity and feasibility of each EAS step in the sequences, explicitly verifying whether the nitration step will yield the desired substitution pattern or if a blocking/protecting group (e.g., sulfonation) is required beforehand. Incorporate the failure reason from previous evaluation that neglecting blocking groups leads to incorrect substitution. Input: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    final_decision_instruction_0_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for regioselectivity and blocking group validation."
    )
    cot_sc_desc_0_2 = {
        "instruction": cot_sc_instruction_0_2,
        "final_decision_instruction": final_decision_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)
    results_stage_0["stage_0.subtask_2"] = results_0_2

    results_stage_1 = {"stage_1.subtask_1": [], "stage_1.subtask_2": [], "stage_1.subtask_3": []}

    for iteration in range(2):
        cot_reflect_instruction_1_1 = (
            "Sub-task 3: Simplify and consolidate the detailed analyses to highlight key reaction order, regioselectivity considerations, and the necessity of blocking groups before nitration, ensuring the failure reason of incorrect directing effects is addressed and avoided. Input: taskInfo, thinking and answer from stage_0.subtask_1 and stage_0.subtask_2"
        )
        critic_instruction_1_1 = (
            "Please review and provide the limitations of provided solutions of simplification and consolidation of synthetic route analyses."
        )
        cot_reflect_desc_1_1 = {
            "instruction": cot_reflect_instruction_1_1,
            "critic_instruction": critic_instruction_1_1,
            "input": [taskInfo, results_stage_0["stage_0.subtask_1"]["thinking"], results_stage_0["stage_0.subtask_1"]["answer"], results_stage_0["stage_0.subtask_2"]["thinking"], results_stage_0["stage_0.subtask_2"]["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_1_1, log_1_1 = await self.reflexion(
            subtask_id="stage_1.subtask_1",
            reflect_desc=cot_reflect_desc_1_1,
            n_repeat=self.max_round
        )
        logs.append(log_1_1)
        results_stage_1["stage_1.subtask_1"].append(results_1_1)

        aggregate_instruction_1_2 = (
            "Sub-task 2: Evaluate each candidate sequence against criteria for high yield, correct substitution pattern, and synthetic feasibility, explicitly incorporating the validation of blocking group strategies and directing effects from stage_0.subtask_2 to avoid repeating past errors. Input: taskInfo, thinking and answer from stage_0.subtask_1 and stage_0.subtask_2"
        )
        aggregate_desc_1_2 = {
            "instruction": aggregate_instruction_1_2,
            "input": [taskInfo] + [r["answer"] for r in results_stage_0.values()] + [r["thinking"] for r in results_stage_0.values()],
            "temperature": 0.0,
            "context": ["user query", "solutions generated from stage_0"]
        }
        results_1_2, log_1_2 = await self.aggregate(
            subtask_id="stage_1.subtask_2",
            aggregate_desc=aggregate_desc_1_2
        )
        logs.append(log_1_2)
        results_stage_1["stage_1.subtask_2"].append(results_1_2)

        cot_agent_instruction_1_3 = (
            "Sub-task 3: Select the best candidate synthetic route based on refined analyses and evaluations, ensuring that the chosen route includes necessary blocking/protecting steps to achieve the target substitution pattern and high yield, thus directly addressing the previous failure reason. Input: taskInfo, thinking and answer from stage_1.subtask_1 and stage_1.subtask_2"
        )
        cot_agent_desc_1_3 = {
            "instruction": cot_agent_instruction_1_3,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_1_2["thinking"], results_1_2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
        }
        results_1_3, log_1_3 = await self.answer_generate(
            subtask_id="stage_1.subtask_3",
            cot_agent_desc=cot_agent_desc_1_3
        )
        logs.append(log_1_3)
        results_stage_1["stage_1.subtask_3"].append(results_1_3)

    results_stage_2 = {}

    review_instruction_2_1 = (
        "Sub-task 1: Apply systematic criteria to assess the correctness, regioselectivity, and overall feasibility of the selected synthetic route, using all prior refined analyses and explicitly confirming that the route avoids the previous logical errors related to directing effects and blocking group omission. Input: taskInfo, thinking and answer from stage_1.subtask_3"
    )
    review_desc_2_1 = {
        "instruction": review_instruction_2_1,
        "input": [taskInfo, results_stage_1["stage_1.subtask_3"][-1]["thinking"], results_stage_1["stage_1.subtask_3"][-1]["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)
    results_stage_2["stage_2.subtask_1"] = results_2_1

    final_answer = await self.make_final_answer(results_stage_2["stage_2.subtask_1"]["thinking"], results_stage_2["stage_2.subtask_1"]["answer"])
    return final_answer, logs
