async def forward_173(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and summarize all given physical data and problem conditions from the query."
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
        "Sub-task 1: Derive the rest masses of the two fragments based on the given mass ratio and total rest mass after fission, using the output from Sub-task 0."
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
        "Sub-task 2: Apply conservation of momentum and energy principles to relate fragment velocities and kinetic energies, based on outputs from Sub-task 1."
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
        "Sub-task 3: Formulate expressions for the relativistic kinetic energy (correct T1) of the more massive fragment, using outputs from Sub-task 2."
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

    cot_instruction_0_4 = (
        "Sub-task 4: Formulate expressions for the classical (non-relativistic) kinetic energy approximation of the more massive fragment, using outputs from Sub-task 2."
    )
    cot_agent_desc_0_4 = {
        "instruction": cot_instruction_0_4,
        "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id="stage_0.subtask_4",
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    cot_instruction_0_5 = (
        "Sub-task 5: Calculate the difference between the relativistic and classical kinetic energy values for the more massive fragment, using outputs from Sub-tasks 3 and 4."
    )
    cot_agent_desc_0_5 = {
        "instruction": cot_instruction_0_5,
        "input": [taskInfo, results_0_3['thinking'], results_0_3['answer'], results_0_4['thinking'], results_0_4['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
    }
    results_0_5, log_0_5 = await self.cot(
        subtask_id="stage_0.subtask_5",
        cot_agent_desc=cot_agent_desc_0_5
    )
    logs.append(log_0_5)

    cot_instruction_1_0 = (
        "Sub-task 0: Combine the derived kinetic energy difference with the given answer choices for evaluation, using output from stage_0.subtask_5."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_instruction_1_0,
        "input": [taskInfo, results_0_5['thinking'], results_0_5['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
    }
    results_1_0, log_1_0 = await self.cot(
        subtask_id="stage_1.subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_agent_instruction_1_1 = (
        "Sub-task 1: Evaluate which answer choice best matches the calculated kinetic energy difference, using output from stage_1.subtask_0."
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
        "Sub-task 0: Validate the physical consistency of the calculated kinetic energy difference and selected answer, using output from stage_1.subtask_1."
    )
    aggregate_desc_2_0 = {
        "instruction": aggregate_instruction_2_0,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_0, log_2_0 = await self.aggregate(
        subtask_id="stage_2.subtask_0",
        aggregate_desc=aggregate_desc_2_0
    )
    logs.append(log_2_0)

    cot_instruction_2_1 = (
        "Sub-task 1: Select the final answer choice that satisfies all problem constraints and physical laws, using output from stage_2.subtask_0."
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

    debate_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the selected answer against the problem context and assumptions, using output from stage_2.subtask_1."
    )
    final_decision_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the selected answer against the problem context and assumptions."
    )
    debate_desc_2_2 = {
        "instruction": debate_instruction_2_2,
        "final_decision_instruction": final_decision_instruction_2_2,
        "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    review_instruction_3_0 = (
        "Sub-task 0: Consolidate the validated answer and supporting reasoning into a clear, formatted final output, using output from stage_2.subtask_2."
    )
    review_desc_3_0 = {
        "instruction": review_instruction_3_0,
        "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id="stage_3.subtask_0",
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    cot_instruction_3_1 = (
        "Sub-task 1: Summarize the key steps and final conclusion for presentation, using output from stage_3.subtask_0."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_instruction_3_1,
        "input": [taskInfo, results_3_0['thinking'], results_3_0['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_0", "answer of stage_3.subtask_0"]
    }
    results_3_1, log_3_1 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1,
        n_repeat=self.max_sc
    )
    logs.append(log_3_1)

    formatter_instruction_3_2 = (
        "Sub-task 2: Format the final answer choice and explanation according to the required output standards, using output from stage_3.subtask_1."
    )
    formatter_desc_3_2 = {
        "instruction": formatter_instruction_3_2,
        "input": [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "format": "short and concise, without explaination"
    }
    results_3_2, log_3_2 = await self.specific_format(
        subtask_id="stage_3.subtask_2",
        formatter_desc=formatter_desc_3_2
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
