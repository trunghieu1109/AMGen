async def forward_163(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}}
    for i in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize given parameters: periods and radial velocity amplitudes for both systems. "
            "Input content: taskInfo (user query)"
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context_desc": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
        logs.append(log_0_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])
        cot_instruction_0_2 = (
            "Sub-task 2: Identify and categorize relationships between orbital parameters and masses using Kepler's laws and radial velocity data. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 iterations"
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["answer"] + loop_results["stage_0.subtask_1"]["thinking"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
        logs.append(log_0_2)
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Calculate total masses of system_1 and system_2 using periods and radial velocity amplitudes, then compute their mass ratio. "
        "Input content: taskInfo, all thinking and answers from stage_0.subtask_1 and stage_0.subtask_2 iterations"
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the mass ratio calculation problem."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_1"]["answer"] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + loop_results["stage_0.subtask_2"]["thinking"],
        "temperature": 0.5,
        "context_desc": ["user query", "answers and thinking of stage_0.subtask_1", "answers and thinking of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_sc_desc_1_1, n_repeat=self.max_sc)
    logs.append(log_1_1)
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Validate the computed mass ratio against physical constraints and problem assumptions. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    critic_instruction_2_1 = (
        "Please review and provide the limitations of the provided solution for the mass ratio validation."
    )
    cot_reflect_desc_2_1 = {
        "instruction": cot_reflect_instruction_2_1,
        "critic_instruction": critic_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.reflexion(subtask_id="stage_2.subtask_1", reflect_desc=cot_reflect_desc_2_1, n_repeat=self.max_round)
    logs.append(log_2_1)
    cot_agent_instruction_3_1 = (
        "Sub-task 1: Consolidate the validated mass ratio and select the closest matching choice from given options. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.answer_generate(subtask_id="stage_3.subtask_1", cot_agent_desc=cot_agent_desc_3_1)
    logs.append(log_3_1)
    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs