async def forward_163(self, taskInfo):
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Extract and summarize all given observational data and relevant assumptions about the two binary star systems, "
        "including orbital periods, radial velocity amplitudes, and physical assumptions (e.g., circular orbits, edge-on inclination). "
        "Ensure clarity on all input parameters and assumptions to prevent ambiguity in later calculations."
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results_1_1, log_1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc_1_1,
        n_repeat=self.max_round
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Determine the mass ratio (q = M2/M1) of the two stars in each system by correctly applying the inverse proportionality "
        "between radial velocity amplitudes and stellar masses. Explicitly derive and verify the mass ratio formula step-by-step to avoid conceptual errors."
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=cot_agent_desc_1_2,
        n_repeat=self.max_round
    )
    logs.append(log_1_2)

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Derive the correct formula for the total mass of each binary system using Kepler's third law combined with the individual radial velocity amplitudes and mass ratios. "
        "Explicitly incorporate the mass ratio into the total mass calculation, avoiding oversimplified assumptions. The derivation should be clear and mathematically rigorous."
    )
    cot_sc_desc_1_3 = {
        'instruction': cot_sc_instruction_1_3,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2",
            "answer of stage_1.subtask_2"
        ]
    }
    results_1_3, log_1_3 = await self.sc_cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_sc_desc_1_3,
        n_repeat=self.max_sc
    )
    logs.append(log_1_3)

    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Perform explicit numeric substitution and step-by-step arithmetic evaluation of the total masses of system_1 and system_2 "
        "using the derived formula from stage_1.subtask_3 and the observational data. Carefully compute intermediate values and the final mass ratio, "
        "showing all steps to ensure correctness."
    )
    cot_reflect_desc_2_1 = {
        'instruction': cot_reflect_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer'], results_1_3['thinking'], results_1_3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2",
            "answer of stage_1.subtask_2",
            "thinking of stage_1.subtask_3",
            "answer of stage_1.subtask_3"
        ]
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    debate_instruction_2_2 = (
        "Sub-task 2: Compare the numerically evaluated mass ratio of system_1 to system_2 against the given multiple-choice options. "
        "Explicitly justify the choice by matching the computed value to the closest option, ensuring no fallback on consensus without numeric verification. "
        "Agents should debate and cross-validate the final answer to avoid previous mistakes."
    )
    debate_desc_2_2 = {
        'instruction': debate_instruction_2_2,
        'context': [
            "user query",
            "thinking of stage_2.subtask_1",
            "answer of stage_2.subtask_1"
        ],
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_2_2, log_2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_2_2,
        n_repeat=self.max_round
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2['thinking'], results_2_2['answer'])
    return final_answer, logs
