async def forward_168(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and summarize the original decay process 2A -> 2B + 2E + 2V, "
        "focusing on particle types, masses, and the continuous total energy spectrum of E particles with endpoint Q. "
        "Clarify the multi-particle final state and implications for energy spectrum continuity to establish a baseline for comparison."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Perform detailed kinematic analysis of the variant decay 2A -> 2B + 2E + M, "
        "listing all final-state particles and counting independent energy-sharing degrees of freedom. "
        "Deduce whether the total energy spectrum of E remains continuous or becomes discrete, explicitly addressing the n-body nature and spectrum continuity."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Evaluate how replacing two V particles with one massless M particle affects the shape and endpoint "
        "of the total energy spectrum of the outgoing E particles, integrating kinematic insights from Subtask 2. "
        "Explicitly reason about modifications to energy distribution and endpoint value Q."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Compare the energy spectrum characteristics (continuity, shape, endpoint) of the original and variant decays "
        "based on previous analyses. Determine the correct description among the given choices, ensuring consistency with nuclear decay kinematics. "
        "Include a final critical reflection to confirm reasoning and choice selection."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
