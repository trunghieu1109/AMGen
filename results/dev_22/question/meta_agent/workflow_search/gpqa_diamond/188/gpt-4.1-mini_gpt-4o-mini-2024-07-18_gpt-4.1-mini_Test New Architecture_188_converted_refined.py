async def forward_188(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Subtask 1: Analyze and classify each effective particle (Magnon, Skyrmion, Pion, Phonon) by explicitly identifying the physical origin and the nature of the symmetry involved, "
        "with focus on distinguishing continuous versus discrete symmetries and clarifying which continuous symmetry, if any, is spontaneously broken. "
        "Explicitly address the misconception that phonons do not arise from spontaneous symmetry breaking by considering phonons as Nambu-Goldstone bosons of broken continuous translational symmetry in crystals. "
        "Document the type of symmetry (global/local, continuous/discrete) broken for each particle to avoid previous errors."
    )

    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ['user query']
    }

    results1, log1 = await self.sc_cot(
        subtask_id='subtask_1',
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Subtask 2: Perform a critical verification and reflexive review of the classifications from Subtask 1, "
        "focusing on challenging and validating key assumptions, especially for borderline or subtle cases such as phonons and skyrmions. "
        "Include domain-expert style debate or reflexion to ensure that each putative Nambu-Goldstone boson indeed corresponds to a broken continuous symmetry, "
        "explicitly confirming phonons as Goldstone modes of broken continuous translation and clarifying the topological versus symmetry-breaking nature of skyrmions. "
        "Catch and correct any misclassifications before final evaluation."
    )

    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }

    results2, log2 = await self.debate(
        subtask_id='subtask_2',
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Subtask 3: Evaluate and compare the verified classifications of the four particles from Subtasks 1 and 2 to identify which particle is not associated with spontaneous symmetry breaking. "
        "Integrate the refined understanding of symmetry types and the verification outcomes to avoid previous misjudgments."
    )

    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2']
    }

    results3, log3 = await self.debate(
        subtask_id='subtask_3',
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Subtask 4: Synthesize the findings from Subtask 3 into a clear, justified conclusion that answers which effective particle is not linked to spontaneous symmetry breaking. "
        "Explicitly reference the nature of the symmetries broken and the reasoning that led to the final answer, ensuring transparency and correctness."
    )

    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }

    results4, log4 = await self.sc_cot(
        subtask_id='subtask_4',
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])

    return final_answer, logs
