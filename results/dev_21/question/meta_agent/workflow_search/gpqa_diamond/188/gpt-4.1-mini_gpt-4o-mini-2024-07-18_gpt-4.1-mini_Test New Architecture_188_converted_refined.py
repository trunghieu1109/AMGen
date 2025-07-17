async def forward_188(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze and classify each given particle (Magnon, Skyrmion, Pion, Phonon) based on whether it arises from spontaneous symmetry breaking, "
        "explicitly identifying the continuous symmetry (if any) that must be spontaneously broken for the particle to exist. "
        "Provide detailed reasoning linking each particle to its associated broken symmetry."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Verify and critically evaluate the classification from Sub-task 1 by explicitly checking the validity of the identified symmetry-particle associations, "
        "with special attention to subtle or ambiguous cases such as phonons and skyrmions. "
        "Re-examine the physical context and the nature of the symmetry breaking involved to catch and correct any misclassifications or unsupported assumptions."
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

    debate_instruction3 = (
        "Sub-task 3: Evaluate and compare the verified classifications from Sub-task 2 to identify which particle is not associated with spontaneous symmetry breaking. "
        "Synthesize the refined and critically reviewed information to reach a well-supported conclusion."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Select the final answer based on the evaluation in Sub-task 3 and provide a clear, detailed explanation supporting why the chosen particle is not linked to spontaneous symmetry breaking. "
        "Explicitly reference the verified symmetry-particle relationships established in earlier subtasks to ensure transparency and robustness of the conclusion."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
