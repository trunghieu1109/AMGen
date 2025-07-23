async def forward_197(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        "instruction": (
            "Sub-task 1: Extract and summarize all given input data: total cobalt concentration, SCN- concentration, "
            "and stability constants β1 to β4. Input content: user query with question and choices."
        ),
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        "instruction": (
            "Sub-task 2: Interpret the meaning of stability constants β1 to β4 as cumulative formation constants for Co(SCN)_n^(2-n) complexes "
            "and define equilibrium expressions for each complex concentration. Input content: user query and results from stage_0.subtask_1 (thinking and answer)."
        ),
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_agent_desc_0_3 = {
        "instruction": (
            "Sub-task 3: Set up the mass balance equation for total cobalt as the sum of free Co(II) and all complex species concentrations, "
            "expressed in terms of free Co(II) and SCN- concentrations and β values. Input content: user query and results from stage_0.subtask_2 (thinking and answer)."
        ),
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)

    cot_agent_desc_0_4 = {
        "instruction": (
            "Sub-task 4: Solve the mass balance equation to find the free Co(II) concentration in the solution. "
            "Input content: user query and results from stage_0.subtask_3 (thinking and answer)."
        ),
        "input": [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(subtask_id="stage_0.subtask_4", cot_agent_desc=cot_agent_desc_0_4)
    logs.append(log_0_4)

    cot_agent_desc_0_5 = {
        "instruction": (
            "Sub-task 5: Calculate the concentrations of each cobalt thiocyanato complex (n=1 to 4) using the free Co(II) concentration, "
            "SCN- concentration, and β values. Input content: user query and results from stage_0.subtask_4 (thinking and answer)."
        ),
        "input": [taskInfo, results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_0_5, log_0_5 = await self.cot(subtask_id="stage_0.subtask_5", cot_agent_desc=cot_agent_desc_0_5)
    logs.append(log_0_5)

    aggregate_desc_1_1 = {
        "instruction": (
            "Sub-task 1: Sum the concentrations of all cobalt species (free Co(II) and complexes) to verify total cobalt concentration consistency. "
            "Input content: user query and results from stage_0.subtask_5 and stage_0.subtask_4 (thinking and answer)."
        ),
        "input": [taskInfo, results_0_5['thinking'], results_0_5['answer'], results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_1_1, log_1_1 = await self.aggregate(subtask_id="stage_1.subtask_1", aggregate_desc=aggregate_desc_1_1)
    logs.append(log_1_1)

    cot_agent_desc_1_2 = {
        "instruction": (
            "Sub-task 2: Calculate the fraction and percentage of the dithiocyanato cobalt(II) complex (n=2) relative to total cobalt concentration. "
            "Input content: user query and results from stage_0.subtask_5 and stage_1.subtask_1 (thinking and answer)."
        ),
        "input": [taskInfo, results_0_5['thinking'], results_0_5['answer'], results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    review_desc_2_1 = {
        "instruction": (
            "Sub-task 1: Validate the calculated percentage of the dithiocyanato complex by checking for physical plausibility and consistency with stability constants and concentrations. "
            "Input content: user query and results from stage_1.subtask_2 (thinking and answer)."
        ),
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.review(subtask_id="stage_2.subtask_1", review_desc=review_desc_2_1)
    logs.append(log_2_1)

    debate_desc_2_2 = {
        "instruction": (
            "Sub-task 2: Select the closest matching answer choice from the provided options based on the calculated percentage. "
            "Input content: user query and results from stage_2.subtask_1 (thinking and answer)."
        ),
        "final_decision_instruction": (
            "Sub-task 2: Select the closest matching answer choice from the provided options based on the calculated percentage."
        ),
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.debate(subtask_id="stage_2.subtask_2", debate_desc=debate_desc_2_2, n_repeat=2)
    logs.append(log_2_2)

    formatter_desc_3_1 = {
        "instruction": (
            "Sub-task 1: Format the selected answer choice into the required output format for final presentation. "
            "Input content: user query and results from stage_2.subtask_2 (thinking and answer)."
        ),
        "format": "short and concise, without explanation",
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_3_1, log_3_1 = await self.specific_format(subtask_id="stage_3.subtask_1", formatter_desc=formatter_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
