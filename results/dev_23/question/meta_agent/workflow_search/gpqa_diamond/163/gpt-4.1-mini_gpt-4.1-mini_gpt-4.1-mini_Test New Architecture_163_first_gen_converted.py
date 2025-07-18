async def forward_163(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize all given observational data and relevant assumptions about the two binary star systems, "
        "including periods, radial velocity amplitudes, and physical assumptions (e.g., circular orbits, edge-on inclination)."
    )
    debate_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=debate_desc_stage0_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Use the radial velocity amplitudes to determine the mass ratio of the two stars in each system by applying the inverse proportionality between RV amplitude and stellar mass."
    )
    debate_desc_stage1_sub1 = {
        'instruction': cot_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage1_subtask1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    cot_sc_instruction_stage1_sub2 = (
        "Sub-task 2: Apply Kepler's third law to relate the orbital periods and total masses of each system, "
        "incorporating the mass ratios from subtask_1 to express total system masses in terms of observable quantities."
    )
    cot_sc_desc_stage1_sub2 = {
        'instruction': cot_sc_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'],
                  results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1",
                    "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.sc_cot(
        subtask_id="stage1_subtask2",
        cot_agent_desc=cot_sc_desc_stage1_sub2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub2)

    cot_sc_instruction_stage2_sub1 = (
        "Sub-task 1: Combine the results from the previous subtasks to calculate the ratio of the total mass of system_1 to that of system_2, "
        "and identify the closest approximate factor from the given choices."
    )
    cot_sc_desc_stage2_sub1 = {
        'instruction': cot_sc_instruction_stage2_sub1,
        'input': [taskInfo,
                  results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
                  results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1",
                    "thinking of stage1_subtask2", "answer of stage1_subtask2"]
    }
    results_stage2_sub1, log_stage2_sub1 = await self.sc_cot(
        subtask_id="stage2_subtask1",
        cot_agent_desc=cot_sc_desc_stage2_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
