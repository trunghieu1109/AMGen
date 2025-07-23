async def forward_152(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": []}

    for iteration in range(3):
        cot_instruction = (
            "Sub-task 0: Analyze the given Michael addition reactions to generate detailed intermediate mechanistic steps and plausible product structures for each reaction (A, B, C), "
            "with context from the user query and previous iterations if any."
        )
        cot_agent_desc = {
            "instruction": cot_instruction,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_cot, log_cot = await self.cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc
        )
        logs.append(log_cot)
        stage0_results["subtask_0"].append(results_cot)

        reflexion_instruction = (
            "Sub-task 1: Refine and consolidate the intermediate mechanistic steps and product proposals by simplifying, resolving tautomeric forms, "
            "and clarifying structural assignments to produce a provisional final output, based on outputs from Sub-task 0 iterations so far."
        )
        critic_instruction = (
            "Please review and provide the limitations of the provided solutions of intermediate mechanistic steps and product proposals, "
            "and suggest improvements or clarifications."
        )
        reflexion_desc = {
            "instruction": reflexion_instruction,
            "critic_instruction": critic_instruction,
            "input": [taskInfo] + [r["thinking"] for r in stage0_results["subtask_0"]] + [r["answer"] for r in stage0_results["subtask_0"]],
            "temperature": 0.0,
            "context": ["user query"] + ["thinking of subtask 0 iteration"]*len(stage0_results["subtask_0"]) + ["answer of subtask 0 iteration"]*len(stage0_results["subtask_0"])
        }
        results_reflexion, log_reflexion = await self.reflexion(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            reflect_desc=reflexion_desc,
            n_repeat=self.max_round
        )
        logs.append(log_reflexion)
        stage0_results["subtask_1"].append(results_reflexion)

    cot_debate_instruction = (
        "Sub-task 0: Evaluate the candidate product sets from the refined outputs against the multiple-choice options to identify the best matching set of products A, B, and C. "
        "Use the refined mechanistic and product proposals from stage_0 to support your evaluation."
    )
    final_decision_instruction = (
        "Sub-task 0: Based on the evaluation, select the best matching choice among the given options A, B, C, or D."
    )
    debate_desc = {
        "instruction": cot_debate_instruction,
        "final_decision_instruction": final_decision_instruction,
        "input": [taskInfo] + [r["thinking"] for r in stage0_results["subtask_1"]] + [r["answer"] for r in stage0_results["subtask_1"]],
        "context": ["user query", "thinking of stage_0 subtask_1 iterations", "answer of stage_0 subtask_1 iterations"],
        "temperature": 0.5
    }
    results_debate, log_debate = await self.debate(
        subtask_id="stage_1_subtask_0",
        debate_desc=debate_desc,
        n_repeat=self.max_round
    )
    logs.append(log_debate)

    aggregate_instruction = (
        "Sub-task 1: Aggregate the evaluation results from the Debate agent and generate a final answer indicating the correct choice among the given options."
    )
    aggregate_desc = {
        "instruction": aggregate_instruction,
        "input": [taskInfo, results_debate["thinking"], results_debate["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1 subtask_0", "answer of stage_1 subtask_0"]
    }
    results_aggregate, log_aggregate = await self.aggregate(
        subtask_id="stage_1_subtask_1",
        aggregate_desc=aggregate_desc
    )
    logs.append(log_aggregate)

    final_answer = await self.make_final_answer(results_aggregate["thinking"], results_aggregate["answer"])
    return final_answer, logs
