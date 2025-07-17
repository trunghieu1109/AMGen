async def forward_181(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the defining attributes of the Mott-Gurney equation, "
        "including the physical parameters (dielectric constant, mobility, device length), the transport regime (space-charge-limited current), "
        "and the assumptions about device characteristics (trap-free, single-carrier, contact type, injection barriers, diffusion and drift currents)."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Evaluate each of the four given statements about the validity of the Mott-Gurney equation by comparing their conditions "
        "(trap states, carrier type, contact type, injection barriers, current components) against the classified attributes and assumptions from subtask_1."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Select the statement that correctly describes the validity conditions of the Mott-Gurney equation based on the evaluation results, "
        "and provide a clear justification for why it is true and why the others are false."
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

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
