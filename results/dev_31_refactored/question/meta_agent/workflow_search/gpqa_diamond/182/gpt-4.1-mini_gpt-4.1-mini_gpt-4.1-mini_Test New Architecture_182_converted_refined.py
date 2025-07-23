async def forward_182(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}, "stage_0.subtask_3": {"thinking": [], "answer": []}}
    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the starting compound structure in detail, confirm substituent positions and connectivity, "
            "and calculate its initial index of hydrogen deficiency (IHD). This subtask sets the baseline for all subsequent transformations. "
            "Input content: taskInfo"
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
        logs.append(log_0_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])

        cot_reflect_instruction_0_2 = (
            "Sub-task 2: Comprehensively evaluate the chemical reactivity of red phosphorus and excess HI on all functional groups present, "
            "explicitly including both carbonyl groups (formyl and carboxylic acid) and all C=C double bonds (ring and vinyl). "
            "This subtask must incorporate the failure reason from previous attempts that overlooked C=C bond reduction/addition, ensuring the full reactivity profile is considered. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1"
        )
        critic_instruction_0_2 = (
            "Please review and provide the limitations of provided solutions of stage_0.subtask_1 to ensure full reactivity profile consideration."
        )
        cot_reflect_desc_0_2 = {
            "instruction": cot_reflect_instruction_0_2,
            "critic_instruction": critic_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query"] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"]
        }
        results_0_2, log_0_2 = await self.reflexion(subtask_id="stage_0.subtask_2", reflect_desc=cot_reflect_desc_0_2, n_repeat=self.max_round)
        logs.append(log_0_2)
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])

        cot_reflect_instruction_0_3 = (
            "Sub-task 3: Based on the full reactivity profile from stage_0.subtask_2, predict and update the intermediate product structure after reaction with red phosphorus and excess HI, "
            "reflecting all reductions and additions including those on C=C bonds. This updated structure will be the basis for IHD calculation. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_2"
        )
        critic_instruction_0_3 = (
            "Please review and provide the limitations of provided solutions of stage_0.subtask_2 to ensure accurate product structure prediction."
        )
        cot_reflect_desc_0_3 = {
            "instruction": cot_reflect_instruction_0_3,
            "critic_instruction": critic_instruction_0_3,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
            "temperature": 0.0,
            "context": ["user query"] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"]
        }
        results_0_3, log_0_3 = await self.reflexion(subtask_id="stage_0.subtask_3", reflect_desc=cot_reflect_desc_0_3, n_repeat=self.max_round)
        logs.append(log_0_3)
        loop_results["stage_0.subtask_3"]["thinking"].append(results_0_3["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results_0_3["answer"])

    cot_instruction_1_1 = (
        "Sub-task 1: Identify and extract all relevant structural features and unsaturations (rings, double bonds, functional groups) from the updated product structure obtained in stage_0.subtask_3, "
        "to prepare for accurate IHD calculation. This subtask must ensure no unsaturation changes are missed, addressing the previous error of ignoring C=C bond transformations. "
        "Input content: taskInfo, all thinking and answers from stage_0.subtask_3"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"],
        "temperature": 0.0,
        "context": ["user query"] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    debate_instruction_2_1 = (
        "Sub-task 1: Evaluate all possible IHD values based on the predicted product structure and extracted unsaturations from stage_1.subtask_1. "
        "Select the most chemically consistent IHD value, explicitly considering the impact of reductions/additions on C=C bonds and carbonyl groups as per the full reactivity profile. "
        "This subtask should incorporate a debate-style reasoning to weigh alternatives and avoid previous oversights. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Select the most consistent and chemically accurate IHD value for the product after reaction with red phosphorus and excess HI."
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "context": ["user query", results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(subtask_id="stage_2.subtask_1", debate_desc=debate_desc_2_1, n_repeat=self.max_round)
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Sub-task 1: Validate the selected IHD value against chemical logic, reaction conditions, and known reactivity of red phosphorus and excess HI, ensuring correctness. "
        "This validation must explicitly reference the failure reason from previous attempts (ignoring C=C bond transformations) to confirm that all relevant transformations have been accounted for and the final IHD is accurate. "
        "Input content: thinking and answer from stage_0.subtask_2 and stage_2.subtask_1"
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + [results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query"] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + [results_2_1["thinking"], results_2_1["answer"]]
    }
    results_3_1, log_3_1 = await self.review(subtask_id="stage_3.subtask_1", review_desc=review_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
