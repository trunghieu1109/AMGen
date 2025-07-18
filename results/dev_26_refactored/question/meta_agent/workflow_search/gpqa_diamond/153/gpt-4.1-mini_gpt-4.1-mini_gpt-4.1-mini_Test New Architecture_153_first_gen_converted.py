async def forward_153(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction0 = (
        "Sub-task 0: Extract and transform raw spectral data (mass spec, IR, 1H NMR) into summarized chemical features relevant for structure elucidation, with context from the given query."
    )
    final_decision_instruction0 = (
        "Sub-task 0: Synthesize and choose the most consistent summary of spectral data features."
    )
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'final_decision_instruction': final_decision_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    cot_sc_instruction1 = (
        "Sub-task 1: Integrate and analyze the extracted spectral features from Sub-task 0 to deduce functional groups, substitution patterns, and molecular formula clues, with context from the query and Sub-task 0 outputs."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent analysis of spectral features."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0", "answer of subtask 0"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Correlate spectral deductions from Sub-tasks 0 and 1 with the given candidate structures to evaluate their consistency with the data."
    )
    critic_instruction2 = (
        "Please review and provide the limitations of provided solutions correlating spectral data with candidate structures."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'critic_instruction': critic_instruction2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Select the most reasonable structural suggestion for the unidentified compound based on conformity to all spectral criteria and chemical reasoning, using outputs from Sub-tasks 1 and 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the final structural suggestion for the unidentified compound."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
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
