async def forward_170(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Explicitly define and clarify the assumption 'only one monobromo derivative is formed' in the context of electrophilic bromination, "
        "specifying how this affects expected regioselectivity and product distribution for ortho/para and meta directors, including halogens and strong EWGs. "
        "This clarification is crucial to avoid inconsistent treatment of substituent directing effects as noted in previous failure. "
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

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze and categorize each substituent (–CH3, –COOC2H5, –Cl, –NO2, –C2H5, –COOH) by their electronic effects (electron-donating or withdrawing) "
        "and classical directing influence (ortho/para or meta), referencing fundamental organic chemistry principles. Avoid lumping meta-directors together without distinction, as this caused errors previously. "
        "Input content: taskInfo, results (thinking and answer) from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Evaluate steric and resonance effects of substituents on para-isomer yield, including how steric hindrance at ortho positions may increase para substitution fraction. "
        "Incorporate literature data or known trends rather than qualitative assumptions alone, to prevent oversimplification errors. "
        "Input content: taskInfo, results (thinking and answer) from stage_0.subtask_2"
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Quantitatively rank the strength of meta-directing substituents (–NO2, –COOC2H5, –COOH) using Hammett sigma_m constants and electrophilic bromination selectivity data from literature. "
        "Derive the relative para-isomer yield order among these meta-directors to correct the previous misordering caused by ignoring resonance withdrawal differences. "
        "Input content: taskInfo, results (thinking and answer) from stage_0.subtask_2"
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_4, log_0_4 = await self.cot(subtask_id="stage_0.subtask_4", cot_agent_desc=cot_agent_desc_0_4)
    logs.append(log_0_4)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            "Sub-task 1: Based on the clarified assumptions and detailed substituent analysis (electronic, steric, and quantitative meta-director ranking), "
            "generate an initial ranking of the six substances by increasing weight fraction of the para-isomer yield. "
            "Ensure the ranking explicitly incorporates the corrected meta-director order and steric effects. "
            "Input content: taskInfo, results (thinking and answer) from stage_0.subtask_3 and stage_0.subtask_4"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_3['thinking'], results_0_3['answer'], results_0_4['thinking'], results_0_4['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_1_2 = (
            "Sub-task 2: Validate and refine the initial ranking by cross-referencing with experimental or literature data on electrophilic bromination para selectivity for similar substituents. "
            "Address any discrepancies or exceptions, and finalize a consistent ranking. This step prevents errors from purely theoretical assumptions and ensures robustness. "
            "Input content: taskInfo, all results (thinking and answer) from all iterations of stage_1.subtask_1"
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_instruction_1_2,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query"] + ["thinking of stage_1.subtask_1 iteration {}".format(i) for i in range(len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]))] + ["answer of stage_1.subtask_1 iteration {}".format(i) for i in range(len(loop_results_stage_1["stage_1.subtask_1"]["answer"]))]
        }
        results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    debate_instruction_2_1 = (
        "Sub-task 1: Compare the validated para-isomer yield ranking with the four given answer choices. "
        "Select the choice that best matches the refined ranking, justifying the selection based on the analysis and data from previous stages. "
        "Input content: taskInfo, results (thinking and answer) from the last iteration of stage_1.subtask_2"
    )
    debate_desc_2_1 = {
        "instruction": debate_instruction_2_1,
        "final_decision_instruction": "Sub-task 1: Select the best matching answer choice for para-isomer yield ranking.",
        "input": [taskInfo, loop_results_stage_1["stage_1.subtask_2"]["thinking"][-1], loop_results_stage_1["stage_1.subtask_2"]["answer"][-1]],
        "context_desc": ["user query", "thinking of stage_1.subtask_2 last iteration", "answer of stage_1.subtask_2 last iteration"],
        "temperature": 0.5
    }
    results_2_1, log_2_1 = await self.debate(subtask_id="stage_2.subtask_1", debate_desc=debate_desc_2_1, n_repeat=self.max_round)
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Sub-task 1: Perform a final consistency check of the selected answer choice against chemical principles, substituent effects, and reaction assumptions. "
        "Provide feedback on the confidence level and any remaining uncertainties to ensure correctness and alignment with expert knowledge. "
        "Input content: taskInfo, results (thinking and answer) from stage_2.subtask_1"
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.review(subtask_id="stage_3.subtask_1", review_desc=review_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
