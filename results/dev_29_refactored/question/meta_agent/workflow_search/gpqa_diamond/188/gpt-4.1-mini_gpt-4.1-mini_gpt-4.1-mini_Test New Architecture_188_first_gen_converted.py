async def forward_188(self, taskInfo):
    logs = []

    cot_instruction_magnon = (
        "Sub-task 0: Analyze Magnon and determine if it is associated with spontaneous symmetry breaking. "
        "Explain the physical origin and symmetry context."
    )
    cot_agent_desc_magnon = {
        'instruction': cot_instruction_magnon,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_magnon, log_magnon = await self.cot(
        subtask_id='stage_0_subtask_0',
        cot_agent_desc=cot_agent_desc_magnon
    )
    logs.append(log_magnon)

    cot_instruction_skyrmion = (
        "Sub-task 1: Analyze Skyrmion and determine if it is associated with spontaneous symmetry breaking. "
        "Explain its topological nature and relation to symmetry breaking."
    )
    cot_agent_desc_skyrmion = {
        'instruction': cot_instruction_skyrmion,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_skyrmion, log_skyrmion = await self.cot(
        subtask_id='stage_0_subtask_1',
        cot_agent_desc=cot_agent_desc_skyrmion
    )
    logs.append(log_skyrmion)

    cot_instruction_pion = (
        "Sub-task 2: Analyze Pion and determine if it is associated with spontaneous symmetry breaking. "
        "Discuss chiral symmetry breaking and pion as pseudo-Goldstone boson."
    )
    cot_agent_desc_pion = {
        'instruction': cot_instruction_pion,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_pion, log_pion = await self.cot(
        subtask_id='stage_0_subtask_2',
        cot_agent_desc=cot_agent_desc_pion
    )
    logs.append(log_pion)

    cot_instruction_phonon = (
        "Sub-task 3: Analyze Phonon and determine if it is associated with spontaneous symmetry breaking. "
        "Explain lattice vibrations and broken translational symmetry."
    )
    cot_agent_desc_phonon = {
        'instruction': cot_instruction_phonon,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ['user query']
    }
    results_phonon, log_phonon = await self.cot(
        subtask_id='stage_0_subtask_3',
        cot_agent_desc=cot_agent_desc_phonon
    )
    logs.append(log_phonon)

    aggregate_instruction = (
        "Sub-task 0: Combine the analyses of Magnon, Skyrmion, Pion, and Phonon to form a consolidated understanding "
        "of their association with spontaneous symmetry breaking."
    )
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results_magnon['answer'], results_skyrmion['answer'], results_pion['answer'], results_phonon['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'Magnon analysis', 'Skyrmion analysis', 'Pion analysis', 'Phonon analysis']
    }
    results_consolidated, log_consolidated = await self.aggregate(
        subtask_id='stage_1_subtask_0',
        aggregate_desc=aggregate_desc
    )
    logs.append(log_consolidated)

    cot_validate_instruction = (
        "Sub-task 0: Validate each particle's association with spontaneous symmetry breaking based on consolidated input. "
        "Provide detailed reasoning for each particle."
    )
    cot_validate_desc = {
        'instruction': cot_validate_instruction,
        'input': [taskInfo, results_consolidated['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'consolidated analysis']
    }
    results_validate, log_validate = await self.cot(
        subtask_id='stage_2_subtask_0',
        cot_agent_desc=cot_validate_desc
    )
    logs.append(log_validate)

    aggregate_select_instruction = (
        "Sub-task 1: Select the particle(s) that are not associated with spontaneous symmetry breaking based on validation."
    )
    aggregate_select_desc = {
        'instruction': aggregate_select_instruction,
        'input': [taskInfo, results_validate['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'validation results']
    }
    results_select, log_select = await self.aggregate(
        subtask_id='stage_2_subtask_1',
        aggregate_desc=aggregate_select_desc
    )
    logs.append(log_select)

    debate_instruction = (
        "Sub-task 2: Evaluate the validity of the selection to ensure correctness and consistency. "
        "Debate the reasoning and confirm the final choice."
    )
    debate_desc = {
        'instruction': debate_instruction,
        'final_decision_instruction': 'Sub-task 2: Confirm the particle not associated with spontaneous symmetry breaking.',
        'input': [taskInfo, results_select['thinking'] if 'thinking' in results_select else '', results_select['answer']],
        'context_desc': ['user query', 'selection reasoning', 'selection answer'],
        'temperature': 0.5
    }
    results_debate, log_debate = await self.debate(
        subtask_id='stage_2_subtask_2',
        debate_desc=debate_desc,
        n_repeat=self.max_round
    )
    logs.append(log_debate)

    formatter_instruction = (
        "Sub-task 0: Format the final answer indicating which particle is not associated with spontaneous symmetry breaking, "
        "with clear explanation."
    )
    formatter_desc = {
        'instruction': formatter_instruction,
        'input': [taskInfo, results_debate['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'debate final answer'],
        'format': 'short and concise, with explanation'
    }
    results_format, log_format = await self.specific_format(
        subtask_id='stage_3_subtask_0',
        formatter_desc=formatter_desc
    )
    logs.append(log_format)

    final_answer = await self.make_final_answer(results_format.get('thinking', ''), results_format.get('answer', ''))

    return final_answer, logs
