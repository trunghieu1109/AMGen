async def forward_177(self, taskInfo):
    logs = []
    loop_results = {"stage_0": {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": [], "subtask_4": []}}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize the given information from the Lagrangian and related definitions to identify relevant physical quantities and their properties."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id="stage_0.subtask_0",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        loop_results["stage_0"]["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Determine the mass dimensions of the fields psi, bar{psi}, and F^{mu nu} using standard QFT conventions, based on the output from Sub-task 0."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        loop_results["stage_0"]["subtask_1"].append(results_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Calculate the mass dimension of the coupling constant kappa by ensuring the interaction Lagrangian has mass dimension 4, based on the output from Sub-task 1."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        loop_results["stage_0"]["subtask_2"].append(results_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Analyze the implications of the mass dimension of kappa on the renormalizability of the theory, based on standard QFT criteria and the output from Sub-task 2."
        )
        cot_agent_desc_0_3 = {
            "instruction": cot_instruction_0_3,
            "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc_0_3
        )
        logs.append(log_0_3)
        loop_results["stage_0"]["subtask_3"].append(results_0_3)

        cot_reflect_instruction_0_4 = (
            "Sub-task 4: Refine and simplify the intermediate results to produce a clear, concise summary of the mass dimension of kappa and the renormalizability conclusion, based on outputs from Sub-tasks 0 to 3."
        )
        critic_instruction_0_4 = (
            "Please review and provide the limitations of provided solutions of mass dimension and renormalizability analysis."
        )
        cot_reflect_desc_0_4 = {
            "instruction": cot_reflect_instruction_0_4,
            "critic_instruction": critic_instruction_0_4,
            "input": [
                taskInfo,
                results_0_0["thinking"], results_0_0["answer"],
                results_0_1["thinking"], results_0_1["answer"],
                results_0_2["thinking"], results_0_2["answer"],
                results_0_3["thinking"], results_0_3["answer"]
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
        results_0_4, log_0_4 = await self.reflexion(
            subtask_id="stage_0.subtask_4",
            reflect_desc=cot_reflect_desc_0_4,
            n_repeat=self.max_round
        )
        logs.append(log_0_4)
        loop_results["stage_0"]["subtask_4"].append(results_0_4)

    cot_agent_instruction_1_0 = (
        "Sub-task 0: Evaluate the four given choices against the refined summary of mass dimension and renormalizability to identify the best matching candidate."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_agent_instruction_1_0,
        "input": [taskInfo] + [res["thinking"] for res in loop_results["stage_0"]["subtask_4"]] + [res["answer"] for res in loop_results["stage_0"]["subtask_4"]],
        "temperature": 0.0,
        "context": ["user query"] + ["refined summaries from stage_0.subtask_4"]
    }
    results_1_0, log_1_0 = await self.answer_generate(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    aggregate_instruction_2_0 = (
        "Sub-task 0: Validate the selected candidate for correctness, consistency, and compliance with QFT principles and problem requirements, based on the output from stage 1.subtask 0."
    )
    aggregate_desc_2_0 = {
        "instruction": aggregate_instruction_2_0,
        "input": [taskInfo, results_1_0["thinking"], results_1_0["answer"]],
        "temperature": 0.0,
        "context": ["user query", "selected candidate from stage_1.subtask_0"]
    }
    results_2_0, log_2_0 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc_2_0
    )
    logs.append(log_2_0)

    final_answer = await self.make_final_answer(results_2_0["thinking"], results_2_0["answer"])
    return final_answer, logs
