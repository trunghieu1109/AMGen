async def forward_173(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []},
                    "stage_0.subtask_2": {"thinking": [], "answer": []},
                    "stage_0.subtask_3": {"thinking": [], "answer": []},
                    "stage_0.subtask_4": {"thinking": [], "answer": []},
                    "stage_0.subtask_5": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_instruction_01 = (
            "Sub-task 1: Extract and clearly define all given physical parameters and initial conditions from the query, "
            "including the initial nucleus rest mass M, rest-mass energy, mass ratio of fragments, and total rest mass loss. "
            "Ensure precise numerical values and units are established for subsequent calculations. "
            "Input content are results (both thinking and answer) from: none (first subtask)."
        )
        cot_agent_desc_01 = {
            "instruction": cot_instruction_01,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_01, log_01 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_01)
        loop_results["stage_0.subtask_1"]["thinking"].append(results_01["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_01["answer"])
        logs.append(log_01)

        cot_instruction_02 = (
            "Sub-task 2: Derive the exact rest masses of the two fragments based on the given mass ratio (2:1) and the total rest mass loss (1%). "
            "Explicitly calculate numerical values for each fragment's rest mass energy, ensuring consistency with initial parameters from subtask_1. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        cot_agent_desc_02 = {
            "instruction": cot_instruction_02,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_02, log_02 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_02)
        loop_results["stage_0.subtask_2"]["thinking"].append(results_02["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_02["answer"])
        logs.append(log_02)

        cot_instruction_03 = (
            "Sub-task 3: Formulate the conservation of momentum and energy equations for the two-fragment system after fission, "
            "expressing the relativistic kinetic energies and momenta of both fragments. Prepare the system of equations needed to solve for the unknown momentum p precisely. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
        )
        cot_agent_desc_03 = {
            "instruction": cot_instruction_03,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_03, log_03 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_03)
        loop_results["stage_0.subtask_3"]["thinking"].append(results_03["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results_03["answer"])
        logs.append(log_03)

        cot_reflect_instruction_04 = (
            "Sub-task 4: Solve numerically for the exact momentum p of the fragments using a precise root-finding method (e.g., Newton-Raphson) applied to the conservation equations from subtask_3. "
            "Avoid crude approximations or integer guesses to prevent significant rounding errors. Provide the momentum value with sufficient significant figures to distinguish MeV-scale differences. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        )
        critic_instruction_04 = (
            "Please review and provide the limitations of provided solutions of momentum p calculation and ensure numerical precision and correctness."
        )
        cot_reflect_desc_04 = {
            "instruction": cot_reflect_instruction_04,
            "critic_instruction": critic_instruction_04,
            "input": [taskInfo] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
        }
        results_04, log_04 = await self.reflexion(subtask_id="stage_0.subtask_4", reflect_desc=cot_reflect_desc_04, n_repeat=self.max_round)
        loop_results["stage_0.subtask_4"]["thinking"].append(results_04["thinking"])
        loop_results["stage_0.subtask_4"]["answer"].append(results_04["answer"])
        logs.append(log_04)

        cot_reflect_instruction_05 = (
            "Sub-task 5: Calculate the velocity and gamma factor (Lorentz factor) of the more massive fragment using the precise momentum p from subtask_4 and its rest mass from subtask_2. "
            "Use these to compute the relativistic kinetic energy T1_rel accurately. Then compute the classical kinetic energy T1_cl using the classical formula (p^2/2m). "
            "Explicitly compare T1_rel and T1_cl, quantify their difference ΔT1, and perform an error analysis to verify that ΔT1 is physically reasonable and consistent with relativistic mechanics. "
            "This subtask addresses previous failures caused by rough approximations and lack of velocity/gamma checks. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2 & stage_0.subtask_4, respectively."
        )
        critic_instruction_05 = (
            "Please review and provide the limitations of provided solutions of kinetic energy calculations and ensure physical consistency and numerical precision."
        )
        cot_reflect_desc_05 = {
            "instruction": cot_reflect_instruction_05,
            "critic_instruction": critic_instruction_05,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + loop_results["stage_0.subtask_4"]["thinking"] + loop_results["stage_0.subtask_4"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_05, log_05 = await self.reflexion(subtask_id="stage_0.subtask_5", reflect_desc=cot_reflect_desc_05, n_repeat=self.max_round)
        loop_results["stage_0.subtask_5"]["thinking"].append(results_05["thinking"])
        loop_results["stage_0.subtask_5"]["answer"].append(results_05["answer"])
        logs.append(log_05)

    debate_instruction_11 = (
        "Sub-task 1: Evaluate the computed difference ΔT1 between relativistic and classical kinetic energies against the provided answer choices. "
        "Identify which choice best matches the calculated ΔT1 value, considering the MeV scale and error margins from subtask_0.subtask_5. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively."
    )
    final_decision_instruction_11 = (
        "Sub-task 1: Evaluate and select the best matching answer choice for the kinetic energy difference ΔT1."
    )
    debate_desc_11 = {
        "instruction": debate_instruction_11,
        "final_decision_instruction": final_decision_instruction_11,
        "input": [taskInfo] + loop_results["stage_0.subtask_5"]["thinking"] + loop_results["stage_0.subtask_5"]["answer"],
        "context": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"],
        "temperature": 0.5
    }
    results_11, log_11 = await self.debate(subtask_id="stage_1.subtask_1", debate_desc=debate_desc_11, n_repeat=self.max_round)
    logs.append(log_11)

    aggregate_instruction_12 = (
        "Sub-task 2: Aggregate the evaluation results from subtask_1 to select the most consistent and accurate answer choice. "
        "Ensure the selection aligns with physical principles and the problem context. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    aggregate_desc_12 = {
        "instruction": aggregate_instruction_12,
        "input": [taskInfo, results_11["thinking"], results_11["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_12, log_12 = await self.aggregate(subtask_id="stage_1.subtask_2", aggregate_desc=aggregate_desc_12)
    logs.append(log_12)

    review_instruction_21 = (
        "Sub-task 1: Review the selected answer for correctness, consistency with physical principles, and alignment with the problem statement. "
        "Check that the numerical precision and reasoning in previous subtasks support the final choice. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    review_desc_21 = {
        "instruction": review_instruction_21,
        "input": [taskInfo, results_12["thinking"], results_12["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_21, log_21 = await self.review(subtask_id="stage_2.subtask_1", review_desc=review_desc_21)
    logs.append(log_21)

    cot_instruction_22 = (
        "Sub-task 2: Perform a final step-by-step reasoning to confirm the validity of the chosen answer and provide a concise justification. "
        "Explicitly address how the refined calculations and error analysis prevent the previous errors of overestimating the kinetic energy difference. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    cot_agent_desc_22 = {
        "instruction": cot_instruction_22,
        "input": [taskInfo, results_21["thinking"], results_21["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_22, log_22 = await self.cot(subtask_id="stage_2.subtask_2", cot_agent_desc=cot_agent_desc_22)
    logs.append(log_22)

    final_answer = await self.make_final_answer(results_22["thinking"], results_22["answer"])
    return final_answer, logs
