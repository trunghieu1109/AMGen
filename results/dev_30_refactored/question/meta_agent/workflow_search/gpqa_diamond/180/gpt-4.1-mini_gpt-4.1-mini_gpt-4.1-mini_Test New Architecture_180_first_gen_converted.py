async def forward_180(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        "instruction": (
            "Sub-task 1: Extract and summarize the given information from the query, focusing on the solar neutrino production, "
            "the pp-III branch stopping scenario, and the neutrino energy bands involved. Input content is the query provided."
        ),
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        "instruction": (
            "Sub-task 2: Analyze the energy spectra of neutrinos produced by the pp-III branch and other branches, "
            "identifying which energy bands (700-800 keV and 800-900 keV) are predominantly influenced by the pp-III branch. "
            "Input content are results (thinking and answer) from stage_0.subtask_1."
        ),
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_agent_desc_0_3 = {
        "instruction": (
            "Sub-task 3: Determine the expected change in neutrino flux in each energy band after the pp-III branch stops, "
            "considering the 8.5-minute neutrino travel time from the Sun to Earth. "
            "Input content are results (thinking and answer) from stage_0.subtask_2."
        ),
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_agent_desc_0_4 = {
        "instruction": (
            "Sub-task 4: Formulate the initial expression or estimate for the flux ratio Flux(band 1) / Flux(band 2) after the pp-III branch stops. "
            "Input content are results (thinking and answer) from stage_0.subtask_3."
        ),
        "input": [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    aggregate_desc_1_1 = {
        "instruction": (
            "Sub-task 1: Combine the summarized information and spectral analysis to consolidate the contributions of the pp-III branch and other branches to each energy band. "
            "Input content are results (thinking and answer) from stage_0.subtask_2 and stage_0.subtask_4."
        ),
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id="stage_1.subtask_1",
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    cot_agent_desc_1_2 = {
        "instruction": (
            "Sub-task 2: Evaluate the impact of stopping the pp-III branch on the total neutrino flux in each energy band, integrating the consolidated data. "
            "Input content are results (thinking and answer) from stage_1.subtask_1."
        ),
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_agent_desc_1_3 = {
        "instruction": (
            "Sub-task 3: Calculate the approximate numerical ratio of Flux(band 1) to Flux(band 2) based on the evaluated impacts. "
            "Input content are results (thinking and answer) from stage_1.subtask_2."
        ),
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    review_desc_2_1 = {
        "instruction": (
            "Sub-task 1: Validate the physical plausibility of the calculated flux ratio by cross-checking with known solar neutrino physics and energy spectra. "
            "Input content are results (thinking and answer) from stage_1.subtask_3."
        ),
        "input": [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    debate_desc_2_2 = {
        "instruction": (
            "Sub-task 2: Select the most consistent flux ratio estimate among possible values, considering the problem's constraints and assumptions. "
            "Input content are results (thinking and answer) from stage_2.subtask_1."
        ),
        "final_decision_instruction": (
            "Sub-task 2: Select the most consistent flux ratio estimate among possible values, considering the problem's constraints and assumptions."
        ),
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round if hasattr(self, 'max_round') else 2
    )
    logs.append(log_2_2)

    cot_agent_desc_2_3 = {
        "instruction": (
            "Sub-task 3: Evaluate the validity of ignoring neutrino flavor oscillations in the context of this flux ratio calculation. "
            "Input content are results (thinking and answer) from stage_2.subtask_2."
        ),
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_2_3, log_2_3 = await self.cot(
        subtask_id="stage_2.subtask_3",
        cot_agent_desc=cot_agent_desc_2_3
    )
    logs.append(log_2_3)

    formatter_desc_3_1 = {
        "instruction": (
            "Sub-task 1: Consolidate the validated flux ratio and reasoning into a clear, concise final answer. "
            "Input content are results (thinking and answer) from stage_2.subtask_3."
        ),
        "input": [taskInfo, results_2_3['thinking'], results_2_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3"],
        "format": "short and concise, without explanation"
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id="stage_3.subtask_1",
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    debate_desc_3_2 = {
        "instruction": (
            "Sub-task 2: Format the final answer to match the provided multiple-choice options and clearly indicate the selected choice. "
            "Input content are results (thinking and answer) from stage_3.subtask_1."
        ),
        "final_decision_instruction": (
            "Sub-task 2: Format the final answer to match the provided multiple-choice options and clearly indicate the selected choice."
        ),
        "input": [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "temperature": 0.5
    }
    results_3_2, log_3_2 = await self.debate(
        subtask_id="stage_3.subtask_2",
        debate_desc=debate_desc_3_2,
        n_repeat=self.max_round if hasattr(self, 'max_round') else 2
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])

    return final_answer, logs
