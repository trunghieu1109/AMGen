async def forward_160(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and summarize all given information from the query related to the electron microscope setup, "
        "vacuum conditions, and mean free path definitions."
    )
    cot_agent_desc_0_0 = {
        "instruction": cot_instruction_0_0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_0, log_0_0 = await self.cot(
        subtask_id="stage_0.subtask_0",
        cot_agent_desc=cot_agent_desc_0_0
    )
    logs.append(log_0_0)

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the relationships between components such as vacuum pressure, electron beam initiation, "
        "and changes in mean free path (λ1 to λ2), based on output from Sub-task 0."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo, results_0_0['thinking'], results_0_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Identify the relevant field of study and theoretical concepts involved (e.g., electron scattering, vacuum physics), "
        "based on output from Sub-task 1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Highlight and document any unclear aspects or assumptions in the problem statement that may affect interpretation of λ2, "
        "based on output from Sub-task 2."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_1_0 = (
        "Sub-task 0: Combine the extracted information and analyses from stage_0 to form a consolidated understanding of the problem scenario, "
        "based on output from stage_0.subtask_3."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_instruction_1_0,
        "input": [taskInfo, results_0_0['thinking'], results_0_0['answer'], results_0_1['thinking'], results_0_1['answer'], results_0_2['thinking'], results_0_2['answer'], results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_0.subtask_0", "thinking and answer of stage_0.subtask_1", "thinking and answer of stage_0.subtask_2", "thinking and answer of stage_0.subtask_3"]
    }
    results_1_0, log_1_0 = await self.cot(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Apply evaluation criteria to compare λ1 and λ2 based on physical principles and the given factor 1.22 in the choices, "
        "using the consolidated understanding from Sub-task 0 of stage 1."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_agent_instruction_1_1,
        "input": [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_2_0 = (
        "Sub-task 0: Validate the consolidated understanding by checking consistency with known physics of electron-gas scattering and vacuum conditions, "
        "based on output from stage_1.subtask_1."
    )
    aggregate_desc_2_0 = {
        "instruction": aggregate_instruction_2_0,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc_2_0
    )
    logs.append(log_2_0)

    cot_instruction_2_1 = (
        "Sub-task 1: Select the most plausible relationship between λ1 and λ2 from the given choices based on validation results, "
        "using output from Sub-task 0 of stage 2."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_2.subtask_0"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of Mike's observation and the physical meaning of the factor 1.22 in the context of mean free path changes, "
        "based on output from Sub-task 1 of stage 2."
    )
    cot_agent_desc_2_2 = {
        "instruction": cot_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    review_instruction_3_0 = (
        "Sub-task 0: Consolidate the validated conclusion about λ2 into a clear, concise final answer, "
        "based on output from stage_2.subtask_2."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_2.subtask_2"]
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id="stage_3.subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    formatter_instruction_3_1 = (
        "Sub-task 1: Format the final answer to explicitly state the relationship between λ2 and λ1, referencing the given choices, "
        "based on output from stage_3.subtask_0."
    )
    formatter_desc_3_1 = {
        "instruction": formatter_instruction_3_1,
        "input": [taskInfo, results_3_0['thinking'], results_3_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_3.subtask_0"],
        "format": "short and concise, without explaination"
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id="stage_3.subtask_1",
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])

    return final_answer, logs
