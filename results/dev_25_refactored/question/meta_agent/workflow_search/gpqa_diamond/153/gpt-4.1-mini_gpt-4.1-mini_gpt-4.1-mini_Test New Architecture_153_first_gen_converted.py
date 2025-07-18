async def forward_153(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and transform the raw spectral data (mass spec, IR, 1H NMR) into interpretable chemical features such as molecular weight, isotopic pattern, functional groups, and proton environments, with context from the given spectral data and choices."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent chemical features extracted from the spectral data."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Integrate the extracted spectral features from Sub-task 1 to deduce key structural characteristics, including presence of chlorine, carboxylic acid group, aromatic substitution pattern, and molecular formula consistency."
    )
    critic_instruction2 = (
        "Please review and provide the limitations of provided solutions of Sub-task 1 and improve the integration of spectral features to deduce structural characteristics."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'critic_instruction': critic_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the four candidate structures against the integrated spectral evidence from Sub-task 2 to identify which structure best fits all data (mass, isotopic pattern, IR, NMR)."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the best fitting structure among the four candidates based on all spectral evidence and reasoning."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
