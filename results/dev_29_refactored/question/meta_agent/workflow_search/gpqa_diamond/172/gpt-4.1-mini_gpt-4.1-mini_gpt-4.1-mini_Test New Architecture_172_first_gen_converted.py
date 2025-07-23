async def forward_172(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": [], "subtask_3": [], "subtask_4": []}

    for iteration in range(3):
        cot_sc_instruction0 = (
            "Sub-task 0: Extract and summarize the given physical parameters and constants relevant to the problem (electron mass, Planck's constant, given Δx, v)."
        )
        cot_sc_desc0 = {
            "instruction": cot_sc_instruction0,
            "final_decision_instruction": "Sub-task 0: Synthesize and choose the most consistent extraction of parameters.",
            "input": [taskInfo],
            "temperature": 0.5,
            "context_desc": ["user query"]
        }
        results0, log0 = await self.sc_cot(
            subtask_id="stage_0.subtask_0",
            cot_agent_desc=cot_sc_desc0,
            n_repeat=self.max_sc
        )
        logs.append(log0)
        stage0_results["subtask_0"].append(results0)

        cot_instruction1 = (
            "Sub-task 1: Apply the Heisenberg uncertainty principle to calculate the minimum uncertainty in momentum Δp using Δx, based on extracted parameters."
        )
        cot_desc1 = {
            "instruction": cot_instruction1,
            "input": [taskInfo, results0["thinking"], results0["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results1, log1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_desc1
        )
        logs.append(log1)
        stage0_results["subtask_1"].append(results1)

        cot_instruction2 = (
            "Sub-task 2: Relate the uncertainty in momentum Δp to the uncertainty in kinetic energy ΔE using the kinetic energy formula E = p^2/(2m) and appropriate approximations."
        )
        cot_desc2 = {
            "instruction": cot_instruction2,
            "input": [taskInfo, results1["thinking"], results1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2, log2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_desc2
        )
        logs.append(log2)
        stage0_results["subtask_2"].append(results2)

        cot_agent_instruction3 = (
            "Sub-task 3: Calculate the numerical value of ΔE using known constants and the results from previous subtasks."
        )
        cot_agent_desc3 = {
            "instruction": cot_agent_instruction3,
            "input": [taskInfo, results2["thinking"], results2["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results3, log3 = await self.answer_generate(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        logs.append(log3)
        stage0_results["subtask_3"].append(results3)

        cot_reflect_instruction4 = (
            "Sub-task 4: Refine and simplify the intermediate results to produce a clear, concise estimate of the minimum uncertainty in energy ΔE."
        )
        critic_instruction4 = (
            "Please review and provide the limitations of provided solutions and refine the estimate of ΔE."
        )
        cot_reflect_desc4 = {
            "instruction": cot_reflect_instruction4,
            "critic_instruction": critic_instruction4,
            "input": [taskInfo, results0["thinking"], results0["answer"], results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4, log4 = await self.reflexion(
            subtask_id="stage_0.subtask_4",
            reflect_desc=cot_reflect_desc4,
            n_repeat=self.max_round
        )
        logs.append(log4)
        stage0_results["subtask_4"].append(results4)

    aggregate_instruction = (
        "Sub-task 0 of stage 1: From refined ΔE estimates generated in stage 0, aggregate these solutions and return the consistent and best estimate for minimum uncertainty in energy ΔE."
    )
    aggregate_desc = {
        "instruction": aggregate_instruction,
        "input": [taskInfo] + [r["thinking"] for r in stage0_results["subtask_4"]] + [r["answer"] for r in stage0_results["subtask_4"]],
        "temperature": 0.0,
        "context_desc": ["user query", "refined ΔE estimates from stage 0"]
    }
    results_agg, log_agg = await self.aggregate(
        subtask_id="stage_1.subtask_0",
        aggregate_desc=aggregate_desc
    )
    logs.append(log_agg)

    select_instruction = (
        "Sub-task 0 of stage 1: Compare the aggregated ΔE value with the provided multiple-choice options and select the best matching candidate."
    )
    select_desc = {
        "instruction": select_instruction,
        "input": [taskInfo, results_agg["thinking"], results_agg["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "aggregated ΔE estimate"]
    }
    results_select, log_select = await self.aggregate(
        subtask_id="stage_1.subtask_1",
        aggregate_desc=select_desc
    )
    logs.append(log_select)

    final_answer = await self.make_final_answer(results_select["thinking"], results_select["answer"])
    return final_answer, logs
