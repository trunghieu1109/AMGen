async def forward_152(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the given Michael addition reactions, identify nucleophiles, electrophiles, and reaction conditions, "
        "and summarize expected reaction mechanisms and intermediate species. Input: taskInfo containing the query with reaction details."
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
        "Sub-task 2: Interpret the provided multiple-choice product options, correlate them with reaction mechanisms, "
        "and identify key structural differences and functional groups relevant to product assignment. "
        "Input: taskInfo and outputs (thinking and answer) from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        debate_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Generate initial provisional assignments of products A, B, and C based on analysis outputs and reasoning about reaction conditions and mechanisms. "
            "Input: taskInfo and outputs (thinking and answer) from stage_0.subtask_2 and all previous outputs of stage_1.subtask_1 and stage_1.subtask_2 if any."
        )
        debate_desc_1_1 = {
            "instruction": debate_instruction_1_1,
            "final_decision_instruction": f"Iteration {iteration+1} - Sub-task 1: Synthesize and choose the best provisional product assignments for A, B, and C.",
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + loop_results_stage_1["stage_1.subtask_2"]["thinking"],
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "previous answers of stage_1.subtask_1", "previous thinkings of stage_1.subtask_1", "previous answers of stage_1.subtask_2", "previous thinkings of stage_1.subtask_2"],
            "temperature": 0.5
        }
        results_1_1, log_1_1 = await self.debate(
            subtask_id="stage_1.subtask_1",
            debate_desc=debate_desc_1_1,
            n_repeat=1
        )
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        debate_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine and consolidate provisional product assignments by evaluating consistency, tautomeric forms, and stereochemical considerations. "
            "Input: taskInfo and outputs (thinking and answer) from stage_1.subtask_1 of all iterations so far."
        )
        debate_desc_1_2 = {
            "instruction": debate_instruction_1_2,
            "final_decision_instruction": f"Iteration {iteration+1} - Sub-task 2: Synthesize and finalize refined product assignments for A, B, and C.",
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_1"]["thinking"],
            "context_desc": ["user query"] + ["answers of stage_1.subtask_1"] * len(loop_results_stage_1["stage_1.subtask_1"]["answer"]) + ["thinkings of stage_1.subtask_1"] * len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]),
            "temperature": 0.5
        }
        results_1_2, log_1_2 = await self.debate(
            subtask_id="stage_1.subtask_2",
            debate_desc=debate_desc_1_2,
            n_repeat=1
        )
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    aggregate_instruction_2_1 = (
        "Sub-task 1: Evaluate all candidate product assignments from analysis and iterative refinement stages, compare against reaction criteria, "
        "and select the choice that best fits the mechanistic and structural evidence. "
        "Input: taskInfo, outputs from stage_0.subtask_2 and all iterations of stage_1.subtask_2."
    )
    aggregate_desc_2_1 = {
        "instruction": aggregate_instruction_2_1,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage_1["stage_1.subtask_2"]["answer"] + loop_results_stage_1["stage_1.subtask_2"]["thinking"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["answers of stage_1.subtask_2"] * len(loop_results_stage_1["stage_1.subtask_2"]["answer"]) + ["thinkings of stage_1.subtask_2"] * len(loop_results_stage_1["stage_1.subtask_2"]["thinking"])
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id="stage_2.subtask_1",
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
