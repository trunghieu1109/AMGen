async def forward_157(self, taskInfo):
    logs = []

    cot_instruction_0_0 = (
        "Sub-task 0: Extract and summarize all given information from the query, including mutation details, protein domains, and molecular phenotypes."
    )
    cot_agent_desc_0_0 = {
        'instruction': cot_instruction_0_0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }
    results_0_0, log_0_0 = await self.cot(
        subtask_id='stage_0.subtask_0',
        cot_agent_desc=cot_agent_desc_0_0
    )
    logs.append(log_0_0)

    cot_instruction_0_1 = (
        "Sub-task 1: Analyze relationships between components such as phosphorylation, dimerization, mutation effects, and dominant-negative mechanisms, based on output from Sub-task 0."
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo, results_0_0['thinking'], results_0_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_0', 'answer of stage_0.subtask_0']
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id='stage_0.subtask_1',
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Identify the relevant field of study and subfields to contextualize the problem and guide reasoning, based on output from Sub-task 1."
    )
    cot_agent_desc_0_2 = {
        'instruction': cot_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id='stage_0.subtask_2',
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Highlight unclear aspects and assumptions in the query that may affect interpretation and answer selection, based on output from Sub-task 2."
    )
    cot_agent_desc_0_3 = {
        'instruction': cot_instruction_0_3,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id='stage_0.subtask_3',
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    aggregate_instruction_1_0 = (
        "Sub-task 0: Combine the summarized information and analysis from stage_0 subtasks to form a coherent consolidated understanding of the problem."
    )
    aggregate_desc_1_0 = {
        'instruction': aggregate_instruction_1_0,
        'input': [taskInfo, 
                  results_0_0['thinking'], results_0_0['answer'],
                  results_0_1['thinking'], results_0_1['answer'],
                  results_0_2['thinking'], results_0_2['answer'],
                  results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions from stage_0 subtasks']
    }
    results_1_0, log_1_0 = await self.aggregate(
        subtask_id='stage_1.subtask_0',
        aggregate_desc=aggregate_desc_1_0
    )
    logs.append(log_1_0)

    cot_instruction_1_1 = (
        "Sub-task 1: Apply evaluation criteria to the consolidated input to preliminarily assess the plausibility of each molecular phenotype option, based on output from Sub-task 0."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo, results_1_0['thinking'], results_1_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_0', 'answer of stage_1.subtask_0']
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id='stage_1.subtask_1',
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_2_0 = (
        "Sub-task 0: Validate the molecular phenotypes against known characteristics of dominant-negative mutations and protein domain functions, based on output from stage_1.subtask_1."
    )
    cot_agent_desc_2_0 = {
        'instruction': cot_instruction_2_0,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id='stage_2.subtask_0',
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    aggregate_instruction_2_1 = (
        "Sub-task 1: Select the molecular phenotype(s) that best fit the dominant-negative mutation Y's expected molecular behavior, based on output from Sub-task 0."
    )
    aggregate_desc_2_1 = {
        'instruction': aggregate_instruction_2_1,
        'input': [taskInfo, results_2_0['thinking'], results_2_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_0', 'answer of stage_2.subtask_0']
    }
    results_2_1, log_2_1 = await self.aggregate(
        subtask_id='stage_2.subtask_1',
        aggregate_desc=aggregate_desc_2_1
    )
    logs.append(log_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Evaluate the validity of the selected phenotype(s) considering the recessive mutation X and heterozygous context of mutation Y, based on output from Sub-task 1."
    )
    cot_agent_desc_2_2 = {
        'instruction': cot_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_2_2, log_2_2 = await self.cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_agent_desc_2_2
    )
    logs.append(log_2_2)

    review_instruction_3_0 = (
        "Sub-task 0: Consolidate the evaluation results into a clear, concise final answer identifying the most likely molecular phenotype caused by mutation Y, based on output from stage_2.subtask_2."
    )
    review_desc_3_0 = {
        'instruction': review_instruction_3_0,
        'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_3_0, log_3_0 = await self.review(
        subtask_id='stage_3.subtask_0',
        review_desc=review_desc_3_0
    )
    logs.append(log_3_0)

    formatter_instruction_3_1 = (
        "Sub-task 1: Format the final answer according to the query's multiple-choice options and provide a brief rationale, based on output from stage_3.subtask_0."
    )
    formatter_desc_3_1 = {
        'instruction': formatter_instruction_3_1,
        'input': [taskInfo, results_3_0['thinking'], results_3_0['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3.subtask_0', 'answer of stage_3.subtask_0'],
        'format': 'short and concise, with rationale'
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id='stage_3.subtask_1',
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])

    return final_answer, logs
