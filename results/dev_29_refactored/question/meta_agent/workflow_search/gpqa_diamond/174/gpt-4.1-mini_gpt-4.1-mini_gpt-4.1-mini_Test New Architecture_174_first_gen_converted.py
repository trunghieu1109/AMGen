async def forward_174(self, taskInfo):
    logs = []
    loop_results = {"stage_0": {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": [], "subtask_4": []}}

    for iteration in range(3):
        cot_instruction_0 = (
            "Sub-task 0: Extract and summarize all given information from the query to establish a clear understanding of the problem context."
        )
        cot_agent_desc_0 = {
            "instruction": cot_instruction_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0, log_0 = await self.cot(
            subtask_id="stage_0.subtask_0",
            cot_agent_desc=cot_agent_desc_0
        )
        logs.append(log_0)
        loop_results["stage_0"]["subtask_0"].append(results_0)

        cot_instruction_1 = (
            "Sub-task 1: Analyze the physical relationships between the spheroidal charge distribution, angular dependence, and wavelength scaling of the radiated power, based on the output from Sub-task 0."
        )
        cot_agent_desc_1 = {
            "instruction": cot_instruction_1,
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_1, log_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_1
        )
        logs.append(log_1)
        loop_results["stage_0"]["subtask_1"].append(results_1)

        cot_instruction_2 = (
            "Sub-task 2: Identify relevant electromagnetic theory concepts (e.g., multipole radiation, angular distribution) that govern the form of f(lambda, theta) and the fraction of maximum power at theta=30 degrees, based on the output from Sub-task 1."
        )
        cot_agent_desc_2 = {
            "instruction": cot_instruction_2,
            "input": [taskInfo, results_1["thinking"], results_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_2, log_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_2
        )
        logs.append(log_2)
        loop_results["stage_0"]["subtask_2"].append(results_2)

        cot_instruction_3 = (
            "Sub-task 3: Formulate possible expressions for the function f(lambda, theta) consistent with the problem’s geometry and physics, and estimate the fraction of maximum power radiated at theta=30 degrees, based on the output from Sub-task 2."
        )
        cot_agent_desc_3 = {
            "instruction": cot_instruction_3,
            "input": [taskInfo, results_2["thinking"], results_2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results_3, log_3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc_3
        )
        logs.append(log_3)
        loop_results["stage_0"]["subtask_3"].append(results_3)

        cot_reflect_instruction_4 = (
            "Sub-task 4: Refine and consolidate the intermediate results to produce a clear provisional output that relates the fraction of A at theta=30 degrees and the lambda-dependence of f, based on the output from Sub-task 3."
        )
        critic_instruction_4 = (
            "Please review and provide the limitations of provided solutions and consolidate the best provisional output for the fraction of A at theta=30 degrees and the lambda-dependence of f."
        )
        cot_reflect_desc_4 = {
            "instruction": cot_reflect_instruction_4,
            "critic_instruction": critic_instruction_4,
            "input": [
                taskInfo,
                results_0["thinking"], results_0["answer"],
                results_1["thinking"], results_1["answer"],
                results_2["thinking"], results_2["answer"],
                results_3["thinking"], results_3["answer"]
            ],
            "temperature": 0.0,
            "context": [
                "user query",
                "thinking of subtask 0", "answer of subtask 0",
                "thinking of subtask 1", "answer of subtask 1",
                "thinking of subtask 2", "answer of subtask 2",
                "thinking of subtask 3", "answer of subtask 3"
            ]
        }
        results_4, log_4 = await self.reflexion(
            subtask_id="stage_0.subtask_4",
            reflect_desc=cot_reflect_desc_4,
            n_repeat=self.max_round
        )
        logs.append(log_4)
        loop_results["stage_0"]["subtask_4"].append(results_4)

    cot_agent_instruction_0 = (
        "Sub-task 0: Evaluate each of the four given choices against the refined intermediate results to determine which best matches the expected fraction and wavelength dependence."
    )
    cot_agent_desc_0 = {
        "instruction": cot_agent_instruction_0,
        "input": [taskInfo] + [res["thinking"] for res in loop_results["stage_0"]["subtask_4"]] + [res["answer"] for res in loop_results["stage_0"]["subtask_4"]],
        "temperature": 0.0,
        "context": ["user query"] + ["thinking of stage_0.subtask_4 iteration"]*3 + ["answer of stage_0.subtask_4 iteration"]*3
    }
    results_eval, log_eval = await self.answer_generate(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_0
    )
    logs.append(log_eval)

    aggregate_instruction_1 = (
        "Sub-task 1: Aggregate the evaluation results and select the best candidate choice that satisfies the problem requirements."
    )
    aggregate_desc_1 = {
        "instruction": aggregate_instruction_1,
        "input": [taskInfo, results_eval["thinking"], results_eval["answer"]],
        "temperature": 0.0,
        "context": ["user query", "evaluation results"]
    }
    results_agg, log_agg = await self.aggregate(
        subtask_id="stage_1.subtask_1",
        aggregate_desc=aggregate_desc_1
    )
    logs.append(log_agg)

    final_answer = await self.make_final_answer(results_agg["thinking"], results_agg["answer"])
    return final_answer, logs
