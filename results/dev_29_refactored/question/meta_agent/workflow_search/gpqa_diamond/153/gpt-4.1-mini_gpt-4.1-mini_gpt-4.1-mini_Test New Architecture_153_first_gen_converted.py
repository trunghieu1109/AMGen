async def forward_153(self, taskInfo):
    logs = []
    loop_results = {}
    for iteration in range(1):
        stage_results = {}

        cot_instruction_0_0 = "Sub-task 0: Extract and summarize mass spectrometry data including molecular ion peaks and isotopic patterns from the given spectral information."
        cot_agent_desc_0_0 = {
            'instruction': cot_instruction_0_0,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id="stage_0.subtask_0",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        stage_results['stage_0.subtask_0'] = results_0_0

        cot_instruction_0_1 = "Sub-task 1: Extract and summarize IR spectral data focusing on functional group identification, using output from Sub-task 0."
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo, results_0_0['thinking'], results_0_0['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_0.subtask_0", "answer of stage_0.subtask_0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        stage_results['stage_0.subtask_1'] = results_0_1

        cot_instruction_0_2 = "Sub-task 2: Extract and summarize 1H NMR data to identify proton environments and substitution patterns, using output from Sub-task 1."
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        stage_results['stage_0.subtask_2'] = results_0_2

        cot_instruction_0_3 = "Sub-task 3: Summarize the given candidate compounds and their structural features relevant to the spectral data, using output from Sub-task 2."
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc_0_3
        )
        logs.append(log_0_3)
        stage_results['stage_0.subtask_3'] = results_0_3

        aggregate_instruction_1_0 = "Sub-task 0: Integrate mass spec, IR, and NMR summaries to form a consolidated spectral profile of the unknown compound, using output from stage_0.subtask_3."
        aggregate_desc_1_0 = {
            'instruction': aggregate_instruction_1_0,
            'input': [taskInfo, results_0_0, results_0_1, results_0_2, results_0_3],
            'temperature': 0.0,
            'context': ["user query", "solutions generated from stage_0"]
        }
        results_1_0, log_1_0 = await self.aggregate(
            subtask_id="stage_1.subtask_0",
            aggregate_desc=aggregate_desc_1_0
        )
        logs.append(log_1_0)
        stage_results['stage_1.subtask_0'] = results_1_0

        cot_instruction_1_1 = "Sub-task 1: Compare consolidated spectral profile against candidate compounds to evaluate compatibility, using output from Sub-task 0 of stage 1."
        cot_agent_desc_1_1 = {
            'instruction': cot_instruction_1_1,
            'input': [taskInfo, results_1_0['thinking'], results_1_0['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_1.subtask_0", "answer of stage_1.subtask_0"]
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        stage_results['stage_1.subtask_1'] = results_1_1

        aggregate_instruction_2_0 = "Sub-task 0: Validate the presence of chlorine by confirming isotopic pattern consistency with candidates, using output from stage_1.subtask_1."
        aggregate_desc_2_0 = {
            'instruction': aggregate_instruction_2_0,
            'input': [taskInfo, results_1_1],
            'temperature': 0.0,
            'context': ["user query", "solutions generated from stage_1.subtask_1"]
        }
        results_2_0, log_2_0 = await self.aggregate(
            subtask_id="stage_2.subtask_0",
            aggregate_desc=aggregate_desc_2_0
        )
        logs.append(log_2_0)
        stage_results['stage_2.subtask_0'] = results_2_0

        debate_instruction_2_1 = "Sub-task 1: Validate functional groups (carboxylic acid vs aldehyde vs ester) using IR and NMR data, based on output from stage_2.subtask_0."
        final_decision_instruction_2_1 = "Sub-task 1: Validate functional groups (carboxylic acid vs aldehyde vs ester) using IR and NMR data."
        debate_desc_2_1 = {
            'instruction': debate_instruction_2_1,
            'final_decision_instruction': final_decision_instruction_2_1,
            'input': [taskInfo, results_2_0['thinking'], results_2_0['answer']],
            'context_desc': ["user query", "thinking of stage_2.subtask_0", "answer of stage_2.subtask_0"],
            'temperature': 0.5
        }
        results_2_1, log_2_1 = await self.debate(
            subtask_id="stage_2.subtask_1",
            debate_desc=debate_desc_2_1,
            n_repeat=self.max_round
        )
        logs.append(log_2_1)
        stage_results['stage_2.subtask_1'] = results_2_1

        debate_instruction_2_2 = "Sub-task 2: Evaluate aromatic substitution pattern (para vs ortho vs meta) using NMR splitting and integration, based on output from stage_2.subtask_1."
        final_decision_instruction_2_2 = "Sub-task 2: Evaluate aromatic substitution pattern (para vs ortho vs meta) using NMR splitting and integration."
        debate_desc_2_2 = {
            'instruction': debate_instruction_2_2,
            'final_decision_instruction': final_decision_instruction_2_2,
            'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
            'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
            'temperature': 0.5
        }
        results_2_2, log_2_2 = await self.debate(
            subtask_id="stage_2.subtask_2",
            debate_desc=debate_desc_2_2,
            n_repeat=self.max_round
        )
        logs.append(log_2_2)
        stage_results['stage_2.subtask_2'] = results_2_2

        cot_instruction_2_3 = "Sub-task 3: Select the candidate compound that best fits all spectral data and validate its structural plausibility, using output from stage_2.subtask_2."
        cot_agent_desc_2_3 = {
            'instruction': cot_instruction_2_3,
            'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
        }
        results_2_3, log_2_3 = await self.cot(
            subtask_id="stage_2.subtask_3",
            cot_agent_desc=cot_agent_desc_2_3
        )
        logs.append(log_2_3)
        stage_results['stage_2.subtask_3'] = results_2_3

        formatter_instruction_3_0 = "Sub-task 0: Consolidate the validated structural suggestion into a clear, concise final answer, using output from stage_2.subtask_3."
        formatter_desc_3_0 = {
            'instruction': formatter_instruction_3_0,
            'input': [taskInfo, results_2_3['thinking'], results_2_3['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_2.subtask_3", "answer of stage_2.subtask_3"],
            'format': 'short and concise, without explaination'
        }
        results_3_0, log_3_0 = await self.specific_format(
            subtask_id="stage_3.subtask_0",
            formatter_desc=formatter_desc_3_0
        )
        logs.append(log_3_0)
        stage_results['stage_3.subtask_0'] = results_3_0

        review_instruction_3_1 = "Sub-task 1: Provide a brief rationale summarizing the spectral evidence supporting the selected structure, using output from stage_3.subtask_0."
        review_desc_3_1 = {
            'instruction': review_instruction_3_1,
            'input': [taskInfo, results_3_0['thinking'], results_3_0['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of stage_3.subtask_0", "answer of stage_3.subtask_0"]
        }
        results_3_1, log_3_1 = await self.review(
            subtask_id="stage_3.subtask_1",
            review_desc=review_desc_3_1
        )
        logs.append(log_3_1)
        stage_results['stage_3.subtask_1'] = results_3_1

        loop_results[iteration] = stage_results

    final_answer = await self.make_final_answer(loop_results[0]['stage_3.subtask_1']['thinking'], loop_results[0]['stage_3.subtask_1']['answer'])
    return final_answer, logs
