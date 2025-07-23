async def forward_196(self, taskInfo):
    logs = []
    loop_results = {}
    for iteration in range(3):
        stage0_results = {}
        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize the given IR and 1H NMR spectral data to identify key functional groups and structural features of compound X."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id="stage0_subtask0",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        stage0_results['subtask_0'] = results_0_0

        cot_instruction_0_1 = (
            "Sub-task 1: Analyze the reaction conditions involving red phosphorus and HI to infer possible chemical transformations on compound X, "
            "based on the summarized spectral data."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0['thinking'], results_0_0['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage0_subtask0", "answer of stage0_subtask0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage0_subtask1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        stage0_results['subtask_1'] = results_0_1

        cot_instruction_0_2 = (
            "Sub-task 2: Interpret the spectral data in the context of the reaction to propose plausible intermediate structures or functional group changes."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage0_subtask2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        stage0_results['subtask_2'] = results_0_2

        stage1_results = {}
        cot_instruction_1_0 = (
            "Sub-task 0: Combine the summarized spectral data and reaction analysis to form a consolidated hypothesis about the structure of the final product."
        )
        cot_agent_desc_1_0 = {
            "instruction": cot_instruction_1_0,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage0_subtask2", "answer of stage0_subtask2"]
        }
        results_1_0, log_1_0 = await self.cot(
            subtask_id="stage1_subtask0",
            cot_agent_desc=cot_agent_desc_1_0
        )
        logs.append(log_1_0)
        stage1_results['subtask_0'] = results_1_0

        cot_agent_instruction_1_1 = (
            "Sub-task 1: Compare the consolidated hypothesis with the given multiple-choice options to narrow down possible final products."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_agent_instruction_1_1,
            "input": [taskInfo, results_1_0['thinking'], results_1_0['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage1_subtask0", "answer of stage1_subtask0"]
        }
        results_1_1, log_1_1 = await self.answer_generate(
            subtask_id="stage1_subtask1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        stage1_results['subtask_1'] = results_1_1

        stage2_results = {}
        aggregate_instruction_2_0 = (
            "Sub-task 0: Validate the chemical plausibility of each candidate product based on spectral data and reaction mechanism."
        )
        aggregate_desc_2_0 = {
            "instruction": aggregate_instruction_2_0,
            "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
            "temperature": 0.0,
            "context": ["user query", "solutions generated from stage1_subtask1"]
        }
        results_2_0, log_2_0 = await self.aggregate(
            subtask_id="stage2_subtask0",
            aggregate_desc=aggregate_desc_2_0
        )
        logs.append(log_2_0)
        stage2_results['subtask_0'] = results_2_0

        cot_instruction_2_1 = (
            "Sub-task 1: Select the candidate(s) that best fit all spectral and reaction criteria."
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo, results_2_0['thinking'], results_2_0['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage2_subtask0", "answer of stage2_subtask0"]
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id="stage2_subtask1",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        stage2_results['subtask_1'] = results_2_1

        cot_instruction_2_2 = (
            "Sub-task 2: Evaluate the validity of the selected candidate(s) to finalize the most likely product."
        )
        cot_agent_desc_2_2 = {
            "instruction": cot_instruction_2_2,
            "input": [taskInfo, results_2_1['thinking'], results_2_1['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage2_subtask1", "answer of stage2_subtask1"]
        }
        results_2_2, log_2_2 = await self.cot(
            subtask_id="stage2_subtask2",
            cot_agent_desc=cot_agent_desc_2_2
        )
        logs.append(log_2_2)
        stage2_results['subtask_2'] = results_2_2

        stage3_results = {}
        review_instruction_3_0 = (
            "Sub-task 0: Consolidate the evaluation results into a clear, concise final answer identifying the product."
        )
        review_desc_3_0 = {
            "instruction": review_instruction_3_0,
            "input": [taskInfo, results_2_2['thinking'], results_2_2['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage2_subtask2", "answer of stage2_subtask2"]
        }
        results_3_0, log_3_0 = await self.review(
            subtask_id="stage3_subtask0",
            review_desc=review_desc_3_0
        )
        logs.append(log_3_0)
        stage3_results['subtask_0'] = results_3_0

        formatter_instruction_3_1 = (
            "Sub-task 1: Format the final answer according to the required output style, including reasoning summary."
        )
        formatter_desc_3_1 = {
            "instruction": formatter_instruction_3_1,
            "input": [taskInfo, results_3_0['thinking'], results_3_0['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage3_subtask0", "answer of stage3_subtask0"],
            "format": "short and concise, without explaination"
        }
        results_3_1, log_3_1 = await self.specific_format(
            subtask_id="stage3_subtask1",
            formatter_desc=formatter_desc_3_1
        )
        logs.append(log_3_1)
        stage3_results['subtask_1'] = results_3_1

        loop_results[iteration] = {
            'stage0': stage0_results,
            'stage1': stage1_results,
            'stage2': stage2_results,
            'stage3': stage3_results
        }

    final_answer = await self.make_final_answer(
        loop_results[2]['stage3']['subtask_1']['thinking'],
        loop_results[2]['stage3']['subtask_1']['answer']
    )

    return final_answer, logs
