async def forward_179(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": [], "subtask_4": [], "subtask_5": []}

    for iteration in range(3):
        cot_sc_instruction0 = (
            "Sub-task 0: Extract and summarize all given physical and geometric information from the problem statement to form a clear problem setup."
        )
        cot_sc_desc0 = {
            "instruction": cot_sc_instruction0,
            "final_decision_instruction": "Sub-task 0: Synthesize and choose the most consistent summary of the problem setup.",
            "input": [taskInfo],
            "temperature": 0.5,
            "context_desc": ["user query"]
        }
        results0, log0 = await self.sc_cot(
            subtask_id="stage_0.subtask_0",
            cot_agent_desc=cot_sc_desc0,
            n_repeat=self.max_sc
        )
        stage0_results["subtask_0"].append(results0)
        logs.append(log0)

        cot_instruction1 = (
            "Sub-task 1: Analyze the electrostatic interactions between the 13 charges, including pairwise potentials and the effect of the 12 charges constrained on the sphere, based on the summary from Sub-task 0."
        )
        cot_agent_desc1 = {
            "instruction": cot_instruction1,
            "input": [taskInfo, results0["thinking"], results0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results1, log1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc1
        )
        stage0_results["subtask_1"].append(results1)
        logs.append(log1)

        cot_instruction2 = (
            "Sub-task 2: Formulate the mathematical expression for the total electrostatic potential energy of the system based on the configuration analyzed in Sub-task 1."
        )
        cot_agent_desc2 = {
            "instruction": cot_instruction2,
            "input": [taskInfo, results1["thinking"], results1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2, log2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc2
        )
        stage0_results["subtask_2"].append(results2)
        logs.append(log2)

        cot_instruction3 = (
            "Sub-task 3: Apply geometric optimization principles to determine the minimum energy configuration of the 12 charges on the sphere, considering symmetry and repulsion, based on the expression from Sub-task 2."
        )
        cot_agent_desc3 = {
            "instruction": cot_instruction3,
            "input": [taskInfo, results2["thinking"], results2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results3, log3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        stage0_results["subtask_3"].append(results3)
        logs.append(log3)

        cot_instruction4 = (
            "Sub-task 4: Calculate the numerical value of the minimum electrostatic potential energy using physical constants and the derived configuration from Sub-task 3."
        )
        cot_agent_desc4 = {
            "instruction": cot_instruction4,
            "input": [taskInfo, results3["thinking"], results3["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4, log4 = await self.cot(
            subtask_id="stage_0.subtask_4",
            cot_agent_desc=cot_agent_desc4
        )
        stage0_results["subtask_4"].append(results4)
        logs.append(log4)

        cot_reflect_instruction5 = (
            "Sub-task 5: Refine and simplify the intermediate results to produce a clear, concise expression and numerical value for the minimum energy, rounded to three decimals, based on the calculation from Sub-task 4."
        )
        critic_instruction5 = (
            "Please review and provide the limitations of provided solutions and suggest improvements for the refined minimum energy calculation."
        )
        cot_reflect_desc5 = {
            "instruction": cot_reflect_instruction5,
            "critic_instruction": critic_instruction5,
            "input": [taskInfo, results0["thinking"], results0["answer"], results4["thinking"], results4["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 4", "answer of subtask 4"]
        }
        results5, log5 = await self.reflexion(
            subtask_id="stage_0.subtask_5",
            reflect_desc=cot_reflect_desc5,
            n_repeat=self.max_round
        )
        stage0_results["subtask_5"].append(results5)
        logs.append(log5)

    best_iteration_index = 0
    best_answer = stage0_results["subtask_5"][0]["answer"]

    cot_agent_instruction1 = (
        "Sub-task 0: Compare the calculated minimum energy value with the provided multiple-choice options to identify the best matching candidate."
    )
    cot_agent_desc = {
        "instruction": cot_agent_instruction1,
        "input": [taskInfo, best_answer],
        "temperature": 0.0,
        "context": ["user query", "minimum energy calculation"]
    }
    results_stage1, log_stage1 = await self.answer_generate(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log_stage1)

    aggregate_instruction2 = (
        "Sub-task 0: Validate the selected candidate answer for correctness, consistency with physical principles, and proper rounding as requested."
    )
    aggregate_desc = {
        "instruction": aggregate_instruction2,
        "input": [taskInfo, results_stage1["thinking"], results_stage1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "selected candidate answer"]
    }
    results_stage2, log_stage2 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage2["thinking"], results_stage2["answer"])

    return final_answer, logs
