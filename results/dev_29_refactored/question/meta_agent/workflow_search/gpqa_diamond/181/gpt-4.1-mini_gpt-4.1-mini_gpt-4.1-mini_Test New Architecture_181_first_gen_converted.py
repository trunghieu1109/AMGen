async def forward_181(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": []}

    for iteration in range(3):
        cot_instruction_0 = (
            "Sub-task 0: Extract and summarize the given information about the Mott-Gurney equation and the conditions described in the choices."
        )
        cot_agent_desc_0 = {
            "instruction": cot_instruction_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0, log_0 = await self.cot(
            subtask_id=f"stage0_subtask0_iter{iteration}",
            cot_agent_desc=cot_agent_desc_0
        )
        logs.append(log_0)
        stage0_results["subtask_0"].append(results_0)

        cot_instruction_1 = (
            "Sub-task 1: Analyze the relationships between the components of the Mott-Gurney equation and the physical assumptions behind the validity conditions, based on the output from Sub-task 0."
        )
        cot_sc_desc_1 = {
            "instruction": cot_instruction_1,
            "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent analysis of the relationships and assumptions.",
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.5,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_1, log_1 = await self.sc_cot(
            subtask_id=f"stage0_subtask1_iter{iteration}",
            cot_agent_desc=cot_sc_desc_1,
            n_repeat=self.max_sc
        )
        logs.append(log_1)
        stage0_results["subtask_1"].append(results_1)

        cot_reflect_instruction_2 = (
            "Sub-task 2: Clarify ambiguous terms and assumptions such as trap-free, contact types, and negligible currents to refine understanding, using outputs from Sub-task 0 and Sub-task 1."
        )
        critic_instruction_2 = (
            "Please review and provide the limitations of the provided clarifications and assumptions regarding the Mott-Gurney equation validity."
        )
        cot_reflect_desc_2 = {
            "instruction": cot_reflect_instruction_2,
            "critic_instruction": critic_instruction_2,
            "input": [taskInfo, results_0["thinking"], results_0["answer"], results_1["thinking"], results_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_2, log_2 = await self.reflexion(
            subtask_id=f"stage0_subtask2_iter{iteration}",
            reflect_desc=cot_reflect_desc_2,
            n_repeat=self.max_round
        )
        logs.append(log_2)
        stage0_results["subtask_2"].append(results_2)

        aggregate_instruction_3 = (
            "Sub-task 3: Consolidate and simplify the intermediate findings from Sub-tasks 0, 1, and 2 into a refined summary for candidate evaluation."
        )
        aggregate_desc_3 = {
            "instruction": aggregate_instruction_3,
            "input": [taskInfo] + [
                stage0_results["subtask_0"][iteration],
                stage0_results["subtask_1"][iteration],
                stage0_results["subtask_2"][iteration]
            ],
            "temperature": 0.0,
            "context": ["user query", "solutions from subtask 0", "solutions from subtask 1", "solutions from subtask 2"]
        }
        results_3, log_3 = await self.aggregate(
            subtask_id=f"stage0_subtask3_iter{iteration}",
            aggregate_desc=aggregate_desc_3
        )
        logs.append(log_3)
        stage0_results["subtask_3"].append(results_3)

    cot_debate_instruction_0 = (
        "Sub-task 0: Evaluate each candidate statement against the refined criteria and physical understanding derived from stage_0 consolidated summaries."
    )
    cot_debate_desc_0 = {
        "instruction": cot_debate_instruction_0,
        "final_decision_instruction": "Sub-task 0: Debate and determine the validity of each candidate statement based on the refined understanding.",
        "input": [taskInfo] + stage0_results["subtask_3"],
        "context": ["user query"] + [f"refined summary iteration {i}" for i in range(3)],
        "temperature": 0.5
    }
    results_stage1_subtask0, log_stage1_subtask0 = await self.debate(
        subtask_id="stage1_subtask0",
        debate_desc=cot_debate_desc_0,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_subtask0)

    aggregate_instruction_1 = (
        "Sub-task 1: Select the candidate statement that best satisfies the validity conditions of the Mott-Gurney equation based on the debate results."
    )
    aggregate_desc_1 = {
        "instruction": aggregate_instruction_1,
        "input": [taskInfo, results_stage1_subtask0],
        "temperature": 0.0,
        "context": ["user query", "debate results"]
    }
    results_stage1_subtask1, log_stage1_subtask1 = await self.aggregate(
        subtask_id="stage1_subtask1",
        aggregate_desc=aggregate_desc_1
    )
    logs.append(log_stage1_subtask1)

    final_answer = await self.make_final_answer(results_stage1_subtask1["thinking"], results_stage1_subtask1["answer"])

    return final_answer, logs
