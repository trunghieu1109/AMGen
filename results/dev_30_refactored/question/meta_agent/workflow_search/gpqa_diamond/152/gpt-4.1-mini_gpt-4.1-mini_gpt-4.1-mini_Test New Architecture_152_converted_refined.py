async def forward_152(self, taskInfo):
    logs = []
    
    cot_agent_desc_stage0 = {
        "instruction": (
            "Sub-task 1: Identify and extract key chemical entities, reaction conditions, and candidate products from the query, "
            "ensuring comprehensive capture of all relevant data for subsequent analysis. Input content is the provided query."
        ),
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_stage0, log_stage0 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0
    )
    logs.append(log_stage0)

    loop_results_stage1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []},
        "stage_1.subtask_3": {"thinking": [], "answer": []},
        "stage_1.subtask_4": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_desc_1 = {
            "instruction": (
                "Sub-task 1: Analyze Michael addition reaction A by applying mechanistic reasoning to propose the most plausible product structure, "
                "explicitly considering regioselectivity, substituent effects, and reaction conditions. Avoid trivial or overly broad conclusions. "
                "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
            ),
            "final_decision_instruction": (
                "Sub-task 1: Synthesize and choose the most consistent answer for Michael addition reaction A."
            ),
            "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"]] + \
                     loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"] + 
                       ["previous thinking of stage_1.subtask_1"] * len(loop_results_stage1["stage_1.subtask_1"]["thinking"])
        }
        results_1, log_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1,
            n_repeat=self.max_sc
        )
        loop_results_stage1["stage_1.subtask_1"]["thinking"].append(results_1["thinking"])
        loop_results_stage1["stage_1.subtask_1"]["answer"].append(results_1["answer"])
        logs.append(log_1)

        cot_sc_desc_2 = {
            "instruction": (
                "Sub-task 2: Analyze Michael addition reaction B by applying mechanistic reasoning to propose the most plausible product structure, "
                "with attention to stereochemistry, tautomerism, and reaction conditions. Ensure reasoning is consistent and detailed. "
                "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
            ),
            "final_decision_instruction": (
                "Sub-task 2: Synthesize and choose the most consistent answer for Michael addition reaction B."
            ),
            "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"]] + \
                     loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"] + 
                       ["previous thinking of stage_1.subtask_2"] * len(loop_results_stage1["stage_1.subtask_2"]["thinking"])
        }
        results_2, log_2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_2,
            n_repeat=self.max_sc
        )
        loop_results_stage1["stage_1.subtask_2"]["thinking"].append(results_2["thinking"])
        loop_results_stage1["stage_1.subtask_2"]["answer"].append(results_2["answer"])
        logs.append(log_2)

        cot_sc_desc_3 = {
            "instruction": (
                "Sub-task 3: Analyze Michael addition reaction C focusing on mechanistic and structural aspects to propose the most plausible product structure, "
                "considering nucleophile identity, reaction conditions, and expected product features. Do not yet finalize tautomeric form assignments. "
                "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
            ),
            "final_decision_instruction": (
                "Sub-task 3: Synthesize and choose the most consistent answer for Michael addition reaction C preliminary analysis."
            ),
            "input": [taskInfo, results_stage0["thinking"], results_stage0["answer"]] + \
                     loop_results_stage1["stage_1.subtask_3"]["thinking"] + loop_results_stage1["stage_1.subtask_3"]["answer"],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"] + 
                       ["previous thinking of stage_1.subtask_3"] * len(loop_results_stage1["stage_1.subtask_3"]["thinking"])
        }
        results_3, log_3 = await self.sc_cot(
            subtask_id="stage_1.subtask_3",
            cot_agent_desc=cot_sc_desc_3,
            n_repeat=self.max_sc
        )
        loop_results_stage1["stage_1.subtask_3"]["thinking"].append(results_3["thinking"])
        loop_results_stage1["stage_1.subtask_3"]["answer"].append(results_3["answer"])
        logs.append(log_3)

        cot_sc_desc_4 = {
            "instruction": (
                "Sub-task 4: Perform a dedicated analysis of keto-enol tautomerism and product nomenclature reconciliation for reaction C. "
                "Confirm that the assigned starting material tautomer matches the product's nomenclature and reaction conditions, explicitly addressing the previous failure of misassigning the enol form when the keto form is mechanistically favored under basic conditions. "
                "Input content are results (both thinking and answer) from: stage_1.subtask_3, respectively."
            ),
            "final_decision_instruction": (
                "Sub-task 4: Synthesize and choose the most consistent tautomeric assignment and nomenclature reconciliation for reaction C."
            ),
            "input": [taskInfo, loop_results_stage1["stage_1.subtask_3"]["thinking"], loop_results_stage1["stage_1.subtask_3"]["answer"]] + \
                     loop_results_stage1["stage_1.subtask_4"]["thinking"] + loop_results_stage1["stage_1.subtask_4"]["answer"],
            "temperature": 0.5,
            "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"] + 
                       ["previous thinking of stage_1.subtask_4"] * len(loop_results_stage1["stage_1.subtask_4"]["thinking"])
        }
        results_4, log_4 = await self.sc_cot(
            subtask_id="stage_1.subtask_4",
            cot_agent_desc=cot_sc_desc_4,
            n_repeat=self.max_sc
        )
        loop_results_stage1["stage_1.subtask_4"]["thinking"].append(results_4["thinking"])
        loop_results_stage1["stage_1.subtask_4"]["answer"].append(results_4["answer"])
        logs.append(log_4)

    cot_reflect_desc_stage2 = {
        "instruction": (
            "Sub-task 1: Evaluate all candidate answers against the refined intermediate results from reactions A, B, and C, including the tautomerism reconciliation for reaction C. "
            "Implement a strict consistency validation step that cross-checks all assigned products for mechanistic, tautomeric, and nomenclature coherence. "
            "If conflicts arise (e.g., keto vs enol form mismatches), flag them for re-evaluation before finalizing the answer. Select the best matching choice accordingly. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1, stage_1.subtask_2, and stage_1.subtask_4, respectively."
        ),
        "critic_instruction": (
            "Please review and provide the limitations of provided solutions of reactions A, B, and C, focusing on tautomerism and nomenclature consistency."
        ),
        "input": [taskInfo] + 
                 loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"] + 
                 loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"] + 
                 loop_results_stage1["stage_1.subtask_4"]["thinking"] + loop_results_stage1["stage_1.subtask_4"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", 
                    "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", 
                    "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"]
    }
    results_stage2, log_stage2 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(
        results_stage2["thinking"],
        results_stage2["answer"]
    )

    return final_answer, logs
