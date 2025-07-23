async def forward_154(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and represent the given operators Px, Py, and Pz as explicit matrices in the Pz eigenbasis, "
        "based on the provided matrix components and physical constants."
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
        "Sub-task 1: Express the given system state vector (eigenstate of Px with eigenvalue -ħ) explicitly in the Pz eigenbasis, "
        "using the information from Sub-task 0."
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
        "Sub-task 2: Calculate the expectation value <Pz> in the given state using the Pz matrix and the state vector, "
        "based on outputs from Sub-task 1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_0['thinking'], results_0_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Calculate the expectation value <Pz^2> in the given state by squaring the Pz matrix and applying it to the state vector, "
        "using outputs from Sub-task 0 and Sub-task 1."
    )
    cot_agent_desc_0_3 = {
        "instruction": cot_instruction_0_3,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer'], results_0_0['thinking'], results_0_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id="stage_0.subtask_3",
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Compute the variance (ΔPz)^2 = <Pz^2> - <Pz>^2 as an intermediate step towards uncertainty, "
        "using results from Sub-task 2 and Sub-task 3."
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer'], results_0_3['thinking'], results_0_3['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    cot_instruction_1_0 = (
        "Sub-task 0: Combine the computed expectation values and variance to determine the uncertainty ΔPz = sqrt(variance), "
        "using the variance computed in stage_0.subtask_4."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_instruction_1_0,
        "input": [taskInfo, results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_1_0, log_1_0 = await self.cot(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Compare the calculated uncertainty ΔPz with the provided multiple-choice options to identify the matching choice, "
        "using the uncertainty computed in stage_1.subtask_0."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_agent_instruction_1_1,
        "input": [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"]
    }
    results_1_1, log_1_1 = await self.answer_generate(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    aggregate_instruction_2_0 = (
        "Sub-task 0: Validate the correctness of the calculated uncertainty value by cross-checking calculations and ensuring physical consistency, "
        "using the answer from stage_1.subtask_1."
    )
    aggregate_desc_2_0 = {
        "instruction": aggregate_instruction_2_0,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "solutions generated from stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc_2_0
    )
    logs.append(log_2_0)

    cot_instruction_2_1 = (
        "Sub-task 1: Select the final answer choice that correctly corresponds to the validated uncertainty value, "
        "based on the validation from stage_2.subtask_0."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the selected answer choice in the context of the problem and quantum mechanical principles, "
        "using the selected answer from stage_2.subtask_1."
    )
    cot_agent_desc_2_2 = {
        "instruction": cot_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    formatter_instruction_3_0 = (
        "Sub-task 0: Format the final answer and supporting reasoning into a clear, concise, and standardized output, "
        "using the evaluation from stage_2.subtask_2."
    )
    formatter_desc_3_0 = {
        "instruction": formatter_instruction_3_0,
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        "format": "short and concise, without explaination"
    }
    results_3_0, log_3_0 = await self.specific_format(
        subtask_id="stage_3.subtask_0",
        formatter_desc=formatter_desc_3_0
    )
    logs.append(log_3_0)

    review_instruction_3_1 = (
        "Sub-task 1: Summarize the key steps and results leading to the final uncertainty value ΔPz for presentation, "
        "based on the formatted output from stage_3.subtask_0."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_3_0['thinking'], results_3_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_0", "answer of stage_3.subtask_0"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])

    return final_answer, logs
