async def forward_177(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the defining features of the interaction Lagrangian, "
        "including the fields involved, their known or assumed canonical mass dimensions, and the role of the tensor sigma_{mu nu}. "
        "Use the provided query and context to support the analysis."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Perform dimensional analysis to determine the mass dimensions of each component in the interaction term "
        "bar{psi} sigma_{mu nu} psi F^{mu nu}, and derive the mass dimension of the coupling constant kappa. "
        "Use the output from Sub-task 1 as context."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Analyze the implications of the mass dimension of kappa on the renormalizability of the theory, "
        "using standard QFT criteria for renormalizability based on operator dimensions. "
        "Use outputs from Sub-task 2 as context."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=cot_agent_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Combine the results from dimensional analysis and renormalizability assessment to select the correct answer choice "
        "from the given options. Use self-consistency chain-of-thought to consider all evidence and finalize the answer."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of subtask 2",
            "answer of subtask 2",
            "thinking of subtask 3",
            "answer of subtask 3"
        ]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
