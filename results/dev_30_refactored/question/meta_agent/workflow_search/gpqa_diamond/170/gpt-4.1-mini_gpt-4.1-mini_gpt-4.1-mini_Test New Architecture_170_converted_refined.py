async def forward_170(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []},
                    "stage_0.subtask_2": {"thinking": [], "answer": []},
                    "stage_0.subtask_3": {"thinking": [], "answer": []},
                    "stage_0.subtask_4": {"thinking": [], "answer": []},
                    "stage_0.subtask_5": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_agent_desc_1 = {
            "instruction": (
                "Sub-task 1: Extract and summarize the given chemical information about the six substances, their substituents, reaction conditions, and assumptions. "
                "Ensure clarity on the nature of the electrophilic substitution and the meaning of 'weight fraction of para-isomer'. "
                "Input content: user query provided in taskInfo."
            ),
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results1, log1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results1["answer"])
        logs.append(log1)

        cot_agent_desc_2 = {
            "instruction": (
                "Sub-task 2: Collect and reference authoritative quantitative data on substituent electronic effects relevant to electrophilic aromatic substitution, "
                "such as Hammett sigma constants (sigma_p) or experimental para-isomer yields, especially focusing on the meta-directing groups NO2, COOH, and COOC2H5. "
                "Input content: results (thinking and answer) from stage_0.subtask_1."
            ),
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results2, log2 = await self.sc_cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_2, n_repeat=self.max_sc)
        loop_results["stage_0.subtask_2"]["thinking"].append(results2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results2["answer"])
        logs.append(log2)

        cot_agent_desc_3 = {
            "instruction": (
                "Sub-task 3: Analyze the electronic effects and directing properties of each substituent to predict regioselectivity trends for electrophilic bromination, "
                "using the quantitative data obtained in stage_0.subtask_2. Avoid mis-ranking COOH and COOC2H5 by basing conclusions on referenced data rather than qualitative intuition. "
                "Input content: results (thinking and answer) from stage_0.subtask_1 and stage_0.subtask_2."
            ),
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results3, log3 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_3)
        loop_results["stage_0.subtask_3"]["thinking"].append(results3["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results3["answer"])
        logs.append(log3)

        cot_agent_desc_4 = {
            "instruction": (
                "Sub-task 4: Estimate the relative weight fractions of the para-isomer formed for each substance based on the regioselectivity analysis from stage_0.subtask_3. "
                "Incorporate steric hindrance and other subtle factors influencing substitution patterns, but ensure electronic effects remain primary. "
                "Correct previous overemphasis on sterics. "
                "Input content: results (thinking and answer) from stage_0.subtask_3."
            ),
            "input": [taskInfo] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
        }
        results4, log4 = await self.cot(subtask_id="stage_0.subtask_4", cot_agent_desc=cot_agent_desc_4)
        loop_results["stage_0.subtask_4"]["thinking"].append(results4["thinking"])
        loop_results["stage_0.subtask_4"]["answer"].append(results4["answer"])
        logs.append(log4)

        aggregate_desc_5 = {
            "instruction": (
                "Sub-task 5: Consolidate and document the final refined para-isomer weight fractions for all substances, preparing data for comparison. "
                "Ensure corrected ranking of meta-directing groups and traceability to quantitative data or chemical principles. "
                "Input content: results (thinking and answer) from stage_0.subtask_4."
            ),
            "input": [taskInfo] + loop_results["stage_0.subtask_4"]["thinking"] + loop_results["stage_0.subtask_4"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results5, log5 = await self.aggregate(subtask_id="stage_0.subtask_5", aggregate_desc=aggregate_desc_5)
        loop_results["stage_0.subtask_5"]["thinking"].append(results5["thinking"])
        loop_results["stage_0.subtask_5"]["answer"].append(results5["answer"])
        logs.append(log5)

    debate_desc_1 = {
        "instruction": (
            "Sub-task 1: Compare the consolidated para-isomer weight fractions across all substances to determine their relative order. "
            "Perform a critical consistency check against known chemical principles and quantitative data to identify and flag contradictions or implausible rankings. "
            "Input content: results (thinking and answer) from all iterations of stage_0.subtask_5."
        ),
        "final_decision_instruction": (
            "Sub-task 1: Perform critical consistency check and identify contradictions or implausible rankings in para-isomer weight fractions."
        ),
        "input": [taskInfo] + loop_results["stage_0.subtask_5"]["thinking"] + loop_results["stage_0.subtask_5"]["answer"],
        "context": ["user query"] + loop_results["stage_0.subtask_5"]["thinking"] + loop_results["stage_0.subtask_5"]["answer"],
        "temperature": 0.5
    }
    results_debate, log_debate = await self.debate(subtask_id="stage_1.subtask_1", debate_desc=debate_desc_1, n_repeat=self.max_round)
    logs.append(log_debate)

    aggregate_desc_2 = {
        "instruction": (
            "Sub-task 2: Select the best candidate ordering of substances by increasing para-isomer weight fraction and justify the choice with explicit reference to quantitative substituent constants, experimental data, and chemical reasoning. "
            "Ensure final answer is chemically plausible and consistent with authoritative data. "
            "Input content: results (thinking and answer) from stage_1.subtask_1."
        ),
        "input": [taskInfo, results_debate["thinking"], results_debate["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_final, log_final = await self.aggregate(subtask_id="stage_1.subtask_2", aggregate_desc=aggregate_desc_2)
    logs.append(log_final)

    final_answer = await self.make_final_answer(results_final["thinking"], results_final["answer"])
    return final_answer, logs
