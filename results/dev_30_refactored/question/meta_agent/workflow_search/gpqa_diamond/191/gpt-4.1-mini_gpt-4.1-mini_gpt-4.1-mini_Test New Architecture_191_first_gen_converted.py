async def forward_191(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize all given physical parameters and geometric relationships from the query. "
        "Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        'instruction': cot_instruction_0_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id='stage_0.subtask_1',
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the electrostatic properties of the spherical conductor with an off-center cavity containing charge +q, including induced charges and shielding effects. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        'instruction': cot_instruction_0_2,
        'input': [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id='stage_0.subtask_2',
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    cot_instruction_0_3 = (
        "Sub-task 3: Formulate the relationship between the position vectors (l, s) and the angle theta to express the effective distance relevant for the electric field calculation at point P. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_2"
    )
    cot_agent_desc_0_3 = {
        'instruction': cot_instruction_0_3,
        'input': [taskInfo, results_0_2['thinking'], results_0_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
    }
    results_0_3, log_0_3 = await self.cot(
        subtask_id='stage_0.subtask_3',
        cot_agent_desc=cot_agent_desc_0_3
    )
    logs.append(log_0_3)

    cot_instruction_0_4 = (
        "Sub-task 4: Apply the method of images or equivalent electrostatic principles to determine the effect of the conductor on the field due to charge +q inside the cavity. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_3"
    )
    cot_agent_desc_0_4 = {
        'instruction': cot_instruction_0_4,
        'input': [taskInfo, results_0_3['thinking'], results_0_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
    }
    results_0_4, log_0_4 = await self.cot(
        subtask_id='stage_0.subtask_4',
        cot_agent_desc=cot_agent_desc_0_4
    )
    logs.append(log_0_4)

    cot_instruction_0_5 = (
        "Sub-task 5: Derive an initial expression for the magnitude of the electric field E at point P outside the conductor based on the above analysis. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_4"
    )
    cot_agent_desc_0_5 = {
        'instruction': cot_instruction_0_5,
        'input': [taskInfo, results_0_4['thinking'], results_0_4['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
    }
    results_0_5, log_0_5 = await self.cot(
        subtask_id='stage_0.subtask_5',
        cot_agent_desc=cot_agent_desc_0_5
    )
    logs.append(log_0_5)

    aggregate_instruction_1_1 = (
        "Sub-task 1: Combine the summarized parameters and derived expressions from stage_0 to form a consolidated formula for the electric field at point P. "
        "Input content: taskInfo, thinking and answer from stage_0.subtask_5"
    )
    aggregate_desc_1_1 = {
        'instruction': aggregate_instruction_1_1,
        'input': [taskInfo, results_0_5['thinking'], results_0_5['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Evaluate the physical consistency of the consolidated formula with respect to boundary conditions and known electrostatic principles. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_1_3 = (
        "Sub-task 3: Compare the consolidated formula with the given multiple-choice options to identify matching expressions. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_2"
    )
    cot_agent_desc_1_3 = {
        'instruction': cot_instruction_1_3,
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id='stage_1.subtask_3',
        cot_agent_desc=cot_agent_desc_1_3
    )
    logs.append(log_1_3)

    review_instruction_2_1 = (
        "Sub-task 1: Validate the physical correctness of each multiple-choice option against the consolidated formula and electrostatic theory. "
        "Input content: taskInfo, thinking and answer from stage_1.subtask_3"
    )
    review_desc_2_1 = {
        'instruction': review_instruction_2_1,
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_2_1
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2: Select the option(s) that satisfy all physical and mathematical criteria for the electric field at point P. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
    )
    debate_desc_2_2 = {
        'instruction': debate_instruction_2_2,
        'final_decision_instruction': debate_instruction_2_2,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'],
        'temperature': 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id='stage_2.subtask_2',
        debate_desc=debate_desc_2_2,
        n_repeat=1
    )
    logs.append(log_2_2)

    cot_instruction_2_3 = (
        "Sub-task 3: Assess the validity of the selected option(s) considering the geometry, charge placement, and vector relationships. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_2"
    )
    cot_agent_desc_2_3 = {
        'instruction': cot_instruction_2_3,
        'input': [taskInfo, results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_2_3, log_2_3 = await self.cot(
        subtask_id='stage_2.subtask_3',
        cot_agent_desc=cot_agent_desc_2_3
    )
    logs.append(log_2_3)

    formatter_instruction_3_1 = (
        "Sub-task 1: Consolidate the evaluation results into a clear, concise final answer indicating the correct expression for the electric field magnitude at point P. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_3"
    )
    formatter_desc_3_1 = {
        'instruction': formatter_instruction_3_1,
        'input': [taskInfo, results_2_3['thinking'], results_2_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_3', 'answer of stage_2.subtask_3']
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id='stage_3.subtask_1',
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    debate_instruction_3_2 = (
        "Sub-task 2: Format the final answer according to the required output style, referencing the correct multiple-choice option. "
        "Input content: taskInfo, thinking and answer from stage_3.subtask_1"
    )
    debate_desc_3_2 = {
        'instruction': debate_instruction_3_2,
        'final_decision_instruction': debate_instruction_3_2,
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1'],
        'temperature': 0.5
    }
    results_3_2, log_3_2 = await self.debate(
        subtask_id='stage_3.subtask_2',
        debate_desc=debate_desc_3_2,
        n_repeat=1
    )
    logs.append(log_3_2)

    final_answer = await self.make_final_answer(results_3_2['thinking'], results_3_2['answer'])
    return final_answer, logs
