async def forward_180(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize the given information from the query, focusing on solar neutrino production, "
        "the hypothetical stopping of the pp-III branch 8.5 minutes ago, and the specified neutrino energy bands (700-800 keV and 800-900 keV). "
        "Ensure clarity on the problem context and assumptions, including ignoring neutrino flavor oscillations. "
        "Input content are results (both thinking and answer) from: taskInfo."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Retrieve and quantify the neutrino flux contributions from each proton-proton chain branch (pp-I, pp-II, pp-III, etc.) specifically within the 700-800 keV and 800-900 keV energy bands using known solar neutrino spectra and branching ratios. "
        "This subtask addresses the critical failure in the previous decomposition where assumptions about spectral dominance were incorrect. "
        "The output must include numerical or proportional flux contributions per branch per band. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    spectrum_lookup_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.spectrum_lookup(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=spectrum_lookup_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Interpret the quantitative spectral data obtained in stage_0.subtask_2 to identify which branches predominantly contribute to each energy band and how stopping the pp-III branch would affect the flux in these bands. "
        "Explicitly avoid assumptions without data support, addressing the previous error of assuming near-zero flux in band 2 after stopping pp-III. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2["thinking"], results_0_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Estimate the expected change in neutrino flux in each energy band after the pp-III branch stops, considering the 8.5-minute neutrino travel time from the Sun to Earth and the quantitative spectral contributions. "
        "This subtask must integrate the spectral interpretation with temporal effects to produce a reasoned flux change estimate. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_3["thinking"], results_0_3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    aggregate_instruction_1_1 = (
        "Sub-task 1: Combine the summarized problem context and the quantitative spectral analysis to consolidate the contributions of the pp-III branch and other branches to each energy band. "
        "This consolidation must explicitly incorporate the numerical flux fractions and their implications for flux changes after stopping pp-III, ensuring no oversimplified assumptions are made. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_0.subtask_4, respectively."
    )
    aggregate_desc_1_1 = {
        "instruction": aggregate_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"], results_0_4["thinking"], results_0_4["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id="stage_1.subtask_1",
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Evaluate the impact of stopping the pp-III branch on the total neutrino flux in each energy band by integrating the consolidated data. "
        "This evaluation must quantitatively assess the relative flux reductions per band and avoid previous errors of assuming complete flux disappearance in band 2. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_1_3 = (
        "Sub-task 3: Calculate the approximate numerical ratio of Flux(band 1) to Flux(band 2) after the pp-III branch stops, based on the evaluated impacts. "
        "The calculation must be grounded in the quantitative spectral data and flux impact evaluation to produce a physically plausible ratio consistent with known solar neutrino physics. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
    )
    cot_agent_desc_1_3 = {
        "instruction": cot_instruction_1_3,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    review_instruction_2_1 = (
        "Sub-task 1: Validate the physical plausibility of the calculated flux ratio by cross-checking with established solar neutrino physics, known energy spectra, and branching ratios. "
        "This validation must explicitly confirm that the ratio aligns with the expected contributions of the 7Be line and other branches, addressing the previous failure of ignoring these spectral details. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_3, respectively."
    )
    review_desc_2_1 = {
        "instruction": review_instruction_2_1,
        "input": [taskInfo, results_1_3["thinking"], results_1_3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2: Select the most consistent and justified flux ratio estimate among possible values, considering the problem's constraints, assumptions, and the validation results. "
        "This selection must avoid overestimations caused by ignoring overlapping spectral contributions. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    debate_desc_2_2 = {
        "instruction": debate_instruction_2_2,
        "final_decision_instruction": "Sub-task 2: Select the most consistent and justified flux ratio estimate for the neutrino flux ratio after stopping pp-III.",
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    cot_instruction_2_3 = (
        "Sub-task 3: Evaluate the validity of ignoring neutrino flavor oscillations in the context of this flux ratio calculation, confirming that this assumption does not significantly affect the final ratio. "
        "This subtask ensures completeness and addresses any residual concerns about flavor effects. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_2, respectively."
    )
    cot_agent_desc_2_3 = {
        "instruction": cot_instruction_2_3,
        "input": [taskInfo, results_2_2["thinking"], results_2_2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_2_3, log_2_3 = await self.cot(
        subtask_id="stage_2.subtask_3",
        cot_agent_desc=cot_agent_desc_2_3
    )
    logs.append(log_2_3)

    formatter_instruction_3_1 = (
        "Sub-task 1: Consolidate the validated flux ratio and the supporting reasoning into a clear, concise final answer. "
        "The summary must explicitly reference the quantitative spectral data and validation steps to justify the selected ratio. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_3, respectively."
    )
    formatter_desc_3_1 = {
        "instruction": formatter_instruction_3_1,
        "input": [taskInfo, results_2_3["thinking"], results_2_3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3"],
        "format": "short and concise, without explanation"
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id="stage_3.subtask_1",
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    debate_instruction_3_2 = (
        "Sub-task 2: Format the final answer to match the provided multiple-choice options and clearly indicate the selected choice, ensuring compliance with the problem's output requirements. "
        "Input content are results (both thinking and answer) from: stage_3.subtask_1, respectively."
    )
    debate_desc_3_2 = {
        "instruction": debate_instruction_3_2,
        "final_decision_instruction": "Sub-task 2: Provide the final formatted answer matching the multiple-choice options.",
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "temperature": 0.5
    }
    results_3_2, log_3_2 = await self.debate(
        subtask_id="stage_3.subtask_2",
        debate_desc=debate_desc_3_2,
        n_repeat=self.max_round
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2["thinking"], results_3_2["answer"])
    return final_answer, logs
