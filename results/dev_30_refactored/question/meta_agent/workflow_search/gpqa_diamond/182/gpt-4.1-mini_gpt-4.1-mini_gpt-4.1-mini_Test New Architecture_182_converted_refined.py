async def forward_182(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []},
                    "stage_0.subtask_2": {"thinking": [], "answer": []},
                    "stage_0.subtask_3": {"thinking": [], "answer": []},
                    "stage_0.subtask_4": {"thinking": [], "answer": []},
                    "stage_0.subtask_5": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Analyze and determine the initial structure and calculate the initial index of hydrogen deficiency (IHD) of 2-formyl-5-vinylcyclohex-3-enecarboxylic acid based on its ring system, substituents, and unsaturation. "
            "This subtask sets the baseline for subsequent analysis. Input: taskInfo"
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_1, log_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results["stage_0.subtask_1"]["thinking"].append(results_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_1["answer"])
        logs.append(log_1)

        debate_instruction_2 = (
            "Sub-task 2: Review and debate the known chemical selectivity and mechanistic effects of red phosphorus and excess HI on different functional groups present in the molecule (aldehydes, carboxylic acids, ring C=C bonds, vinyl C=C bonds). "
            "Explicitly address the failure in previous reasoning where all double bonds were assumed fully reduced. Use literature precedent and mechanistic rationale to determine which bonds are reduced, which become iodinated, and which remain unchanged. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        debate_desc_2 = {
            "instruction": debate_instruction_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
            "temperature": 0.5,
            "final_decision_instruction": (
                "Sub-task 2: Review and debate the known chemical selectivity and mechanistic effects of red phosphorus and excess HI on different functional groups present in the molecule, and synthesize the most consistent mechanistic understanding."
            )
        }
        results_2, log_2 = await self.debate(
            subtask_id="stage_0.subtask_2",
            debate_desc=debate_desc_2,
            n_repeat=self.max_round
        )
        loop_results["stage_0.subtask_2"]["thinking"].append(results_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_2["answer"])
        logs.append(log_2)

        cot_instruction_3 = (
            "Sub-task 3: Apply the mechanistic conclusions from the debate to predict the structure of the product formed after reaction with red phosphorus and excess HI. "
            "Determine the transformations of each functional group and unsaturation accordingly, ensuring no assumptions of full saturation of ring double bonds unless justified. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_0.subtask_1, respectively."
        )
        cot_agent_desc_3 = {
            "instruction": cot_instruction_3,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_3, log_3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results["stage_0.subtask_3"]["thinking"].append(results_3["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results_3["answer"])
        logs.append(log_3)

        cot_instruction_4 = (
            "Sub-task 4: Calculate the expected index of hydrogen deficiency (IHD) of the product based on the predicted product structure and transformations identified. "
            "Explicitly incorporate the mechanistic selectivity insights to avoid undercounting residual unsaturation, especially the ring double bond. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        )
        cot_agent_desc_4 = {
            "instruction": cot_instruction_4,
            "input": [taskInfo] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
        }
        results_4, log_4 = await self.cot(
            subtask_id="stage_0.subtask_4",
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results["stage_0.subtask_4"]["thinking"].append(results_4["thinking"])
        loop_results["stage_0.subtask_4"]["answer"].append(results_4["answer"])
        logs.append(log_4)

        aggregate_instruction_5 = (
            "Sub-task 5: Summarize and refine the intermediate reasoning steps and the calculated IHD to produce a coherent intermediate output. "
            "Ensure that the summary explicitly notes the mechanistic selectivity considerations and their impact on the final IHD calculation to prevent repeating previous errors. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4, respectively."
        )
        aggregate_desc_5 = {
            "instruction": aggregate_instruction_5,
            "input": [taskInfo] + loop_results["stage_0.subtask_4"]["thinking"] + loop_results["stage_0.subtask_4"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_5, log_5 = await self.aggregate(
            subtask_id="stage_0.subtask_5",
            aggregate_desc=aggregate_desc_5
        )
        loop_results["stage_0.subtask_5"]["thinking"].append(results_5["thinking"])
        loop_results["stage_0.subtask_5"]["answer"].append(results_5["answer"])
        logs.append(log_5)

    debate_instruction_6 = (
        "Stage 1 Sub-task 1: Evaluate the refined intermediate outputs from stage_0.subtask_5 to identify the most plausible IHD value among the given choices. "
        "Consider the mechanistic reasoning and IHD calculations carefully to avoid previous mistakes of undercounting unsaturation. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively."
    )
    debate_desc_6 = {
        "instruction": debate_instruction_6,
        "input": [taskInfo] + loop_results["stage_0.subtask_5"]["thinking"] + loop_results["stage_0.subtask_5"]["answer"],
        "context_desc": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"],
        "temperature": 0.5,
        "final_decision_instruction": (
            "Stage 1 Sub-task 1: Evaluate and debate the best candidate IHD value for the product based on refined intermediate outputs."
        )
    }
    results_6, log_6 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_6,
        n_repeat=self.max_round
    )
    logs.append(log_6)

    aggregate_instruction_7 = (
        "Stage 1 Sub-task 2: Aggregate the evaluations from the debate to select the best candidate IHD value for the product, ensuring consensus and consistency with chemical reasoning and reaction conditions. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    aggregate_desc_7 = {
        "instruction": aggregate_instruction_7,
        "input": [taskInfo, results_6["thinking"], results_6["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_7, log_7 = await self.aggregate(
        subtask_id="stage_1.subtask_2",
        aggregate_desc=aggregate_desc_7
    )
    logs.append(log_7)

    review_instruction_8 = (
        "Stage 2 Sub-task 1: Review the selected IHD candidate for chemical correctness and consistency with the reaction conditions, structural changes, and mechanistic insights. "
        "Explicitly check that the ring double bond and other unsaturations are correctly accounted for, addressing the key failure in previous reasoning. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    review_desc_8 = {
        "instruction": review_instruction_8,
        "input": [taskInfo, results_7["thinking"], results_7["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_8, log_8 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_8
    )
    logs.append(log_8)

    cot_instruction_9 = (
        "Stage 2 Sub-task 2: Perform a final step-by-step reasoning to confirm or reject the selected IHD value, producing a validation outcome that explicitly references the mechanistic selectivity and IHD calculation steps to ensure correctness. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    cot_agent_desc_9 = {
        "instruction": cot_instruction_9,
        "input": [taskInfo, results_8["thinking"], results_8["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_9, log_9 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_9
    )
    logs.append(log_9)

    final_answer = await self.make_final_answer(results_9["thinking"], results_9["answer"])
    return final_answer, logs
