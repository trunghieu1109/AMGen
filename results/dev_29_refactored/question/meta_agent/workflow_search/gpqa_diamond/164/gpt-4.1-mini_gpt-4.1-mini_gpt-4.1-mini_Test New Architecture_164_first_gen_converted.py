async def forward_164(self, taskInfo):
    logs = []
    loop_results_stage0 = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": []}

    for iteration in range(3):
        cot_sc_instruction0 = (
            "Sub-task 0: Extract and summarize the given information from the query to identify key facts about the polymerization process and catalyst systems."
        )
        cot_sc_desc0 = {
            "instruction": cot_sc_instruction0,
            "final_decision_instruction": "Sub-task 0: Synthesize and choose the most consistent summary of the polymerization process and catalyst systems.",
            "input": [taskInfo],
            "temperature": 0.5,
            "context_desc": ["user query"]
        }
        results0, log0 = await self.sc_cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_sc_desc0,
            n_repeat=self.max_sc
        )
        logs.append(log0)
        loop_results_stage0["subtask_0"].append(results0)

        cot_instruction1 = (
            "Sub-task 1: Analyze relationships between components such as catalysts, activators, and polymer structure to understand the implications of each statement, based on the summary from Sub-task 0."
        )
        cot_desc1 = {
            "instruction": cot_instruction1,
            "input": [taskInfo, results0["thinking"], results0["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results1, log1 = await self.cot(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            cot_agent_desc=cot_desc1
        )
        logs.append(log1)
        loop_results_stage0["subtask_1"].append(results1)

        cot_reflect_instruction2 = (
            "Sub-task 2: Identify and clarify ambiguous or missing information relevant to the problem, such as catalyst specifics and industrial implementation details, based on outputs from Sub-tasks 0 and 1."
        )
        critic_instruction2 = (
            "Please review and provide the limitations of provided solutions regarding catalyst specifics and industrial implementation details."
        )
        cot_reflect_desc2 = {
            "instruction": cot_reflect_instruction2,
            "critic_instruction": critic_instruction2,
            "input": [taskInfo, results0["thinking"], results0["answer"], results1["thinking"], results1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2, log2 = await self.reflexion(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            reflect_desc=cot_reflect_desc2,
            n_repeat=self.max_round
        )
        logs.append(log2)
        loop_results_stage0["subtask_2"].append(results2)

        aggregate_instruction3 = (
            "Sub-task 3: Refine the intermediate outputs by simplifying and enhancing the analysis to produce a clearer, structured understanding of the problem context, aggregating outputs from Sub-tasks 0, 1, and 2 of this iteration."
        )
        aggregate_desc3 = {
            "instruction": aggregate_instruction3,
            "input": [taskInfo, results0["thinking"], results0["answer"], results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "solutions from subtasks 0,1,2"]
        }
        results3, log3 = await self.aggregate(
            subtask_id=f"stage_0_subtask_3_iter_{iteration}",
            aggregate_desc=aggregate_desc3
        )
        logs.append(log3)
        loop_results_stage0["subtask_3"].append(results3)

    refined_outputs = loop_results_stage0["subtask_3"]

    debate_instruction0 = (
        "Sub-task 0: Evaluate each of the four statements against the refined understanding from stage_0 to determine which best fits the known scientific and industrial context."
    )
    final_decision_instruction0 = (
        "Sub-task 0: Debate and select the most correct statement regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system."
    )
    debate_desc0 = {
        "instruction": debate_instruction0,
        "final_decision_instruction": final_decision_instruction0,
        "input": [taskInfo] + [item["thinking"] for item in refined_outputs] + [item["answer"] for item in refined_outputs],
        "context_desc": ["user query"] + ["thinking of stage_0_subtask_3 iteration"]*3 + ["answer of stage_0_subtask_3 iteration"]*3,
        "temperature": 0.5
    }
    results_stage1_subtask0, log_stage1_subtask0 = await self.debate(
        subtask_id="stage_1_subtask_0",
        debate_desc=debate_desc0,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_subtask0)

    aggregate_instruction1 = (
        "Sub-task 1: Rank the four statements based on correctness, feasibility, and alignment with industrial practice and catalyst chemistry, using the debate results."
    )
    aggregate_desc1 = {
        "instruction": aggregate_instruction1,
        "input": [taskInfo, results_stage1_subtask0["thinking"], results_stage1_subtask0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1_subtask_0", "answer of stage_1_subtask_0"]
    }
    results_stage1_subtask1, log_stage1_subtask1 = await self.aggregate(
        subtask_id="stage_1_subtask_1",
        aggregate_desc=aggregate_desc1
    )
    logs.append(log_stage1_subtask1)

    cot_instruction0_stage2 = (
        "Sub-task 0: Validate the selected best candidate statement by cross-checking with known literature, industrial data, and catalyst chemistry principles."
    )
    cot_desc0_stage2 = {
        "instruction": cot_instruction0_stage2,
        "input": [taskInfo, results_stage1_subtask1["thinking"], results_stage1_subtask1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1_subtask_1", "answer of stage_1_subtask_1"]
    }
    results_stage2_subtask0, log_stage2_subtask0 = await self.cot(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_desc0_stage2
    )
    logs.append(log_stage2_subtask0)

    review_instruction1_stage2 = (
        "Sub-task 1: Produce a final assessment outcome confirming the correctness and consistency of the selected statement."
    )
    review_desc1_stage2 = {
        "instruction": review_instruction1_stage2,
        "input": [taskInfo, results_stage2_subtask0["thinking"], results_stage2_subtask0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2_subtask_0", "answer of stage_2_subtask_0"]
    }
    results_stage2_subtask1, log_stage2_subtask1 = await self.review(
        subtask_id="stage_2_subtask_1",
        review_desc=review_desc1_stage2
    )
    logs.append(log_stage2_subtask1)

    final_answer = await self.make_final_answer(results_stage2_subtask1["thinking"], results_stage2_subtask1["answer"])

    return final_answer, logs
