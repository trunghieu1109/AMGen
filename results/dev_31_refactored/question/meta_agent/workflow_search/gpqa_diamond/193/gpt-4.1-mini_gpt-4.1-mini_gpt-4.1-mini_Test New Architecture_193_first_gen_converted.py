async def forward_193(self, taskInfo):
    logs = []
    loop_results_stage1 = {"stage_1.subtask_1": {"thinking": [], "answer": []}, "stage_1.subtask_2": {"thinking": [], "answer": []}}

    cot_instruction_0_1 = (
        "Sub-task 1: Identify all possible spin configurations and compute their corresponding energies using E = -J(S1S2 + S1S3 + S2S3). "
        "Input content: problem query with energy formula and spin values."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    for iteration in range(2):
        cot_instruction_1_1 = (
            "Sub-task 1: Enumerate the degeneracies of each distinct energy level by grouping spin configurations with identical energies. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and all previous iterations of stage_1.subtask_2 answers and thinkings."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]] + loop_results_stage1["stage_1.subtask_2"]["answer"] + loop_results_stage1["stage_1.subtask_2"]["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "previous thinkings of stage_1.subtask_2", "previous answers of stage_1.subtask_2"]
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_1_2 = (
            "Sub-task 2: Express the partition function Z as a sum over energy levels weighted by their degeneracies and exponentials of -βE. "
            "Input content: results (thinking and answer) from stage_1.subtask_1."
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_instruction_1_2,
            "input": [taskInfo] + loop_results_stage1["stage_1.subtask_1"]["thinking"] + loop_results_stage1["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinkings of stage_1.subtask_1", "answers of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
        logs.append(log_1_2)
        loop_results_stage1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Calculate the explicit exponential terms e^{-βE} for each energy level and sum them according to degeneracies to obtain the partition function expression. "
        "Input content: results (thinking and answer) from stage_1.subtask_2."
    )
    critic_instruction_2_1 = (
        "Please review and provide the limitations of provided solutions of calculating explicit exponential terms and summing them for partition function."
    )
    cot_reflect_desc_2_1 = {
        "instruction": cot_reflect_instruction_2_1,
        "critic_instruction": critic_instruction_2_1,
        "input": [taskInfo] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinkings of stage_1.subtask_2", "answers of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.reflexion(subtask_id="stage_2.subtask_1", reflect_desc=cot_reflect_desc_2_1, n_repeat=2)
    logs.append(log_2_1)

    debate_instruction_3_1 = (
        "Sub-task 1: Simplify and consolidate the partition function expression and compare it against the provided candidate choices to select the best matching formula. "
        "Input content: results (thinking and answer) from stage_1.subtask_2."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Select the best matching partition function formula from the given choices based on simplification and comparison."
    )
    debate_desc_3_1 = {
        "instruction": debate_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
        "context_desc": ["user query", "thinkings of stage_1.subtask_2", "answers of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_3_1, log_3_1 = await self.debate(subtask_id="stage_3.subtask_1", debate_desc=debate_desc_3_1, n_repeat=2)
    logs.append(log_3_1)

    review_instruction_4_1 = (
        "Sub-task 1: Verify the correctness and consistency of the selected partition function expression with the problem's physical and mathematical constraints. "
        "Input content: results (thinking and answer) from stage_2.subtask_1 and stage_3.subtask_1."
    )
    review_desc_4_1 = {
        "instruction": review_instruction_4_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"], results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_4_1, log_4_1 = await self.review(subtask_id="stage_4.subtask_1", review_desc=review_desc_4_1)
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
