async def forward_183(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": []}

    for iteration in range(3):
        cot_instruction_0 = (
            "Sub-task 0: Extract and summarize the given information about starting material, target molecule, "
            "and reaction options to generate an initial structured sequence of intermediate synthetic steps."
        )
        cot_agent_desc_0 = {
            "instruction": cot_instruction_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0, log_0 = await self.cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0
        )
        logs.append(log_0)
        stage0_results["subtask_0"].append(results_0)

        cot_reflect_instruction_1 = (
            "Sub-task 1: Analyze the directing effects, reaction types, and order of reagents to refine and consolidate "
            "the intermediate steps into a coherent synthetic pathway."
        )
        critic_instruction_1 = (
            "Please review and provide the limitations of provided solutions of Sub-task 0 and suggest improvements."
        )
        cot_reflect_desc_1 = {
            "instruction": cot_reflect_instruction_1,
            "critic_instruction": critic_instruction_1,
            "input": [taskInfo, results_0["thinking"], results_0["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_1, log_1 = await self.reflexion(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            reflect_desc=cot_reflect_desc_1,
            n_repeat=self.max_round
        )
        logs.append(log_1)
        stage0_results["subtask_1"].append(results_1)

        aggregate_instruction_2 = (
            "Sub-task 2: Simplify and enhance the intermediate synthetic sequence to produce a refined output that clearly "
            "shows the rationale and expected outcome of each step, aggregating solutions from Sub-tasks 0 and 1."
        )
        aggregate_desc_2 = {
            "instruction": aggregate_instruction_2,
            "input": [taskInfo] + [
                (stage0_results["subtask_0"][-1]["thinking"], stage0_results["subtask_0"][-1]["answer"]),
                (stage0_results["subtask_1"][-1]["thinking"], stage0_results["subtask_1"][-1]["answer"])
            ],
            "temperature": 0.0,
            "context": ["user query", "solutions generated from subtask 0 and 1"]
        }
        results_2, log_2 = await self.aggregate(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            aggregate_desc=aggregate_desc_2
        )
        logs.append(log_2)
        stage0_results["subtask_2"].append(results_2)

    cot_debate_instruction_0 = (
        "Sub-task 0: Evaluate each candidate reaction sequence against criteria such as regioselectivity, yield potential, "
        "and logical order of functional group transformations to determine the best synthetic route."
    )
    final_decision_instruction_0 = (
        "Sub-task 0: Select the best candidate sequence that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene."
    )
    debate_desc_0 = {
        "instruction": cot_debate_instruction_0,
        "final_decision_instruction": final_decision_instruction_0,
        "input": [taskInfo, stage0_results["subtask_2"][-1]["thinking"], stage0_results["subtask_2"][-1]["answer"]],
        "context": ["user query", "thinking of stage_0_subtask_2", "answer of stage_0_subtask_2"],
        "temperature": 0.5
    }
    results_3, log_3 = await self.debate(
        subtask_id="stage_1_subtask_0",
        debate_desc=debate_desc_0,
        n_repeat=self.max_round
    )
    logs.append(log_3)

    cot_answer_instruction_1 = (
        "Sub-task 1: Select the best candidate sequence that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene, "
        "based on the evaluation from the debate."
    )
    cot_answer_desc_1 = {
        "instruction": cot_answer_instruction_1,
        "input": [taskInfo, results_3["thinking"], results_3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1_subtask_0", "answer of stage_1_subtask_0"]
    }
    results_4, log_4 = await self.answer_generate(
        subtask_id="stage_1_subtask_1",
        cot_agent_desc=cot_answer_desc_1
    )
    logs.append(log_4)

    final_answer = await self.make_final_answer(results_4["thinking"], results_4["answer"])
    return final_answer, logs
