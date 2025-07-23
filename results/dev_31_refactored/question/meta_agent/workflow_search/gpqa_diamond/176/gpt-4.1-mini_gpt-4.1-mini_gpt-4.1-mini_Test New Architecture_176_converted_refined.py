async def forward_176(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize all given physical parameters, assumptions, and observational data from the query, "
        "ensuring clarity on radius, mass, observed peak wavelengths, radial velocities, and black body radiation assumption. "
        "Input content are results (both thinking and answer) from: taskInfo."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            "Sub-task 1: Correct the observed peak wavelengths for Doppler shift to obtain intrinsic peak wavelengths for both stars, "
            "using their radial velocities. Then compute the intrinsic effective temperatures T1 and T2 from Wien's displacement law. "
            "This step explicitly addresses the previous failure of assuming equal observed peak wavelengths imply equal temperatures without Doppler correction. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, *loop_results_stage_1["stage_1.subtask_1"]["thinking"], *loop_results_stage_1["stage_1.subtask_1"]["answer"], results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_instruction_1_2 = (
            "Sub-task 2: Analyze the relationships between the stars' physical parameters (radius, mass, intrinsic temperatures) "
            "and derive the formula for the luminosity ratio L1/L2, incorporating the corrected temperatures from stage_1.subtask_1. "
            "Ensure the temperature difference due to Doppler correction is properly included in the luminosity calculation. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_1.subtask_1, respectively."
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_instruction_1_2,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + [results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    cot_instruction_2_1 = (
        "Sub-task 1: Calculate the numerical value of the luminosity ratio L1/L2 using the derived formula and given scaling factors. "
        "Simplify the expression to a final approximate factor, ensuring all Doppler corrections and physical relations are correctly applied. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    cot_reflect_instruction_3_1 = (
        "Sub-task 1: Evaluate the calculated luminosity ratio for physical correctness and consistency with black body radiation laws and Doppler shift effects. "
        "This validation step explicitly aims to prevent the previous error of ignoring Doppler corrections and to ensure the final ratio is physically plausible. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    critic_instruction_3_1 = (
        "Please review and provide the limitations of provided solutions of the luminosity ratio calculation, "
        "checking for physical correctness and consistency with black body radiation and Doppler effects."
    )
    cot_reflect_desc_3_1 = {
        "instruction": cot_reflect_instruction_3_1,
        "critic_instruction": critic_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.reflexion(subtask_id="stage_3.subtask_1", reflect_desc=cot_reflect_desc_3_1, n_repeat=self.max_round)
    logs.append(log_3_1)

    cot_agent_instruction_4_1 = (
        "Sub-task 1: Compare the validated luminosity ratio with the provided multiple-choice options and select the closest matching factor as the final answer. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_2.subtask_1 & stage_3.subtask_1, respectively."
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_agent_instruction_4_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"], results_3_1["thinking"], results_3_1["answer"], results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_4_1, log_4_1 = await self.answer_generate(subtask_id="stage_4.subtask_1", cot_agent_desc=cot_agent_desc_4_1)
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
