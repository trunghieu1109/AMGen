async def forward_162(self, taskInfo):
    logs = []

    cot_agent_desc_0_1 = {
        "instruction": (
            "Sub-task 1: Extract and summarize all given data from the query, including mass of Fe(OH)3, solution volume, acid concentration, and temperature. "
            "Input content are results (both thinking and answer) from: none."
        ),
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_agent_desc_0_2 = {
        "instruction": (
            "Sub-task 2: Calculate the number of moles of Fe(OH)3 based on its molar mass and given mass. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        ),
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_0_2)
    logs.append(log_0_2)

    cot_agent_desc_0_3 = {
        "instruction": (
            "Sub-task 3: Determine the stoichiometric amount of H+ ions required to dissolve Fe(OH)3 completely, based on the dissolution reaction. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
        ),
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_0_3)
    logs.append(log_0_3)

    cot_agent_desc_0_4 = {
        "instruction": (
            "Sub-task 4: Calculate the minimum volume of 0.1 M acid needed to provide the required moles of H+ ions. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        ),
        "input": [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(subtask_id="stage_0.subtask_4", cot_agent_desc=cot_agent_desc_0_4)
    logs.append(log_0_4)

    cot_agent_desc_0_5 = {
        "instruction": (
            "Sub-task 5: Estimate the pH of the resulting solution after acid addition and dilution to 100 cm³, considering excess acid concentration. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4, respectively."
        ),
        "input": [taskInfo, results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_0_5, log_0_5 = await self.cot(subtask_id="stage_0.subtask_5", cot_agent_desc=cot_agent_desc_0_5)
    logs.append(log_0_5)

    aggregate_desc_1_1 = {
        "instruction": (
            "Sub-task 1: Combine calculated acid volume and pH values into a single consolidated result for comparison. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4 & stage_0.subtask_5, respectively."
        ),
        "input": [taskInfo, results_0_4['thinking'], results_0_4['answer'], results_0_5['thinking'], results_0_5['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
    }
    results_1_1, log_1_1 = await self.aggregate(subtask_id="stage_1.subtask_1", aggregate_desc=aggregate_desc_1_1)
    logs.append(log_1_1)

    cot_agent_desc_1_2 = {
        "instruction": (
            "Sub-task 2: Compare the consolidated results with the multiple-choice options to identify the closest matching pair. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
        ),
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
    logs.append(log_1_2)

    review_desc_2_1 = {
        "instruction": (
            "Sub-task 1: Validate the selected acid volume and pH against chemical plausibility and stoichiometric correctness. "
            "Input content are results (both thinking and answer) from: stage_1.subtask_2, respectively."
        ),
        "input": [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.review(subtask_id="stage_2.subtask_1", review_desc=review_desc_2_1)
    logs.append(log_2_1)

    debate_desc_2_2 = {
        "instruction": (
            "Sub-task 2: Evaluate the validity of the selected multiple-choice option as the final answer. "
            "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
        ),
        "final_decision_instruction": "Sub-task 2: Evaluate and select the best final answer based on chemical correctness and stoichiometry.",
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(subtask_id="stage_2.subtask_2", debate_desc=debate_desc_2_2, n_repeat=self.max_round)
    logs.append(log_2_2)

    formatter_desc_3_1 = {
        "instruction": (
            "Sub-task 1: Format the validated acid volume and pH into the required output format matching the multiple-choice style. "
            "Input content are results (both thinking and answer) from: stage_2.subtask_2, respectively."
        ),
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        "format": "short and concise, without explanation"
    }
    results_3_1, log_3_1 = await self.specific_format(subtask_id="stage_3.subtask_1", formatter_desc=formatter_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
