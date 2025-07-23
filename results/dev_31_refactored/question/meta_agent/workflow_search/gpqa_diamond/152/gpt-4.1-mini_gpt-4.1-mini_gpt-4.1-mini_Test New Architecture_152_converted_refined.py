async def forward_152(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the given Michael addition reactions to identify nucleophiles, electrophiles, reaction conditions, "
        "and summarize expected reaction mechanisms and intermediate species, explicitly noting assumptions about reagent roles. "
        "Incorporate feedback to avoid misassigning unusual reagents such as 1-(cyclohex-1-en-1-yl)piperidine. "
        "Input content are results (both thinking and answer) from: taskInfo."
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
        "Sub-task 2: Explicitly verify the molecular roles of all reagents, especially unusual ones like 1-(cyclohex-1-en-1-yl)piperidine, "
        "distinguishing between enamine precursors, enolates, catalysts, or nucleophiles. This subtask addresses the previous failure of misassigning reagent roles and ensures correct mechanistic interpretation. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []},
        "stage_1.subtask_3": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: List and analyze possible keto and enol tautomeric forms for each Michael addition product candidate "
            "under the given reaction conditions (solvent, acid/base, temperature). Apply known equilibria stepwise to determine the predominant tautomeric form for each product, "
            "explicitly addressing the previous error of assuming incorrect tautomeric forms. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_1"]["thinking"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        debate_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Generate initial provisional assignments of products A, B, and C based on the verified reagent roles "
            "and predominant tautomeric forms. Use mechanistic reasoning and reaction conditions to correlate product structures with the multiple-choice options, "
            "ensuring no propagation of previous tautomer or reagent role errors. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1."
        )
        debate_desc_1_2 = {
            "instruction": debate_instruction_1_2,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"],
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
            "temperature": 0.5
        }
        results_1_2, log_1_2 = await self.debate(subtask_id="stage_1.subtask_2", debate_desc=debate_desc_1_2, n_repeat=self.max_round)
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

        debate_instruction_1_3 = (
            f"Iteration {iteration+1} - Sub-task 3: Refine and consolidate provisional product assignments by evaluating consistency with reaction mechanisms, "
            "tautomeric equilibria, stereochemical considerations, and substitution patterns. Use outputs from previous subtasks and prior iteration if applicable. "
            "This subtask explicitly prevents repeating earlier mistakes by cross-checking all assumptions. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_2."
        )
        debate_desc_1_3 = {
            "instruction": debate_instruction_1_3,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
            "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
            "temperature": 0.5
        }
        results_1_3, log_1_3 = await self.debate(subtask_id="stage_1.subtask_3", debate_desc=debate_desc_1_3, n_repeat=self.max_round)
        logs.append(log_1_3)
        loop_results_stage_1["stage_1.subtask_3"]["thinking"].append(results_1_3["thinking"])
        loop_results_stage_1["stage_1.subtask_3"]["answer"].append(results_1_3["answer"])

    aggregate_instruction_2_1 = (
        "Sub-task 1: Evaluate all candidate product assignments from analysis and iterative refinement stages, compare against mechanistic and structural evidence, "
        "and select the choice that best fits the reaction criteria. Inputs include outputs from stage_0.subtask_2 and stage_1.subtask_3. "
        "Ensure final selection is robust against previous errors in reagent role and tautomer assumptions. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_1.subtask_3."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]] + loop_results_stage_1["stage_1.subtask_3"]["thinking"] + loop_results_stage_1["stage_1.subtask_3"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_2_1, log_2_1 = await self.aggregate(subtask_id="stage_2.subtask_1", aggregate_desc=aggregate_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1["thinking"], results_2_1["answer"])
    return final_answer, logs
