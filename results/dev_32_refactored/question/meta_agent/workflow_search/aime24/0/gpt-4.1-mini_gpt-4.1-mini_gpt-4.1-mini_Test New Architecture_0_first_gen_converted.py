async def forward_0(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}}

    for i in range(3):
        cot_instruction_0_1 = (
            "Stage 0, Sub-task 1: Formulate equations relating walking speed s, coffee break time t, and total times for given speeds based on the problem statement. "
            "Input content are results (both thinking and answer) from: none."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.6,
            "context_desc": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Stage 0, Sub-task 2: Express walking times excluding coffee break and rewrite equations to isolate variables. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 from all iterations."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_0.subtask_1 all iterations", "answer of stage_0.subtask_1 all iterations"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
        logs.append(log_0_2)

    cot_instruction_1_1 = (
        "Stage 1, Sub-task 1: Solve the system of equations from stage_0 to find numerical values of s and t. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2 from all iterations."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2 all iterations", "answer of stage_0.subtask_2 all iterations"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_reflect_instruction_2_1 = (
        "Stage 2, Sub-task 1: Calculate the walking time for 9 km at speed s + 0.5 km/h and add coffee break time t to find total time. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1."
    )
    critic_instruction_2_1 = (
        "Stage 2, Sub-task 1, Criticism: Please review and provide the limitations of provided solutions of calculating total walking time at speed s + 0.5 km/h including coffee break time t."
    )
    cot_reflect_desc_2_1 = {
        "instruction": cot_reflect_instruction_2_1,
        "critic_instruction": critic_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    cot_reflect_instruction_3_1 = (
        "Stage 3, Sub-task 1: Check consistency and correctness of the computed total time with problem constraints. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_2.subtask_1."
    )
    critic_instruction_3_1 = (
        "Stage 3, Sub-task 1, Criticism: Please review and provide feedback on the consistency and correctness of the computed total time with problem constraints."
    )
    cot_reflect_desc_3_1 = {
        "instruction": cot_reflect_instruction_3_1,
        "critic_instruction": critic_instruction_3_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc_3_1,
        n_repeat=self.max_round
    )
    logs.append(log_3_1)

    cot_agent_instruction_4_1 = (
        "Stage 4, Sub-task 1: Convert the total time into minutes and present the final answer clearly. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_3.subtask_1."
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_agent_instruction_4_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results_4_1, log_4_1 = await self.answer_generate(
        subtask_id="stage_4.subtask_1",
        cot_agent_desc=cot_agent_desc_4_1
    )
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
