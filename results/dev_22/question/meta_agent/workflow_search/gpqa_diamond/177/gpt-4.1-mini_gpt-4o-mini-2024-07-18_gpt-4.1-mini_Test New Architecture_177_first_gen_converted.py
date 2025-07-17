async def forward_177(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the defining features of the problem, including the given Lagrangian, "
        "the fields involved, their known or standard mass dimensions, and the definition of sigma_{mu nu}. "
        "Use the provided query context to debate and clarify these features."
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
        "Sub-task 2: Determine the canonical mass dimensions of the fields psi and F^{mu nu}, "
        "and confirm the dimension of the Lagrangian density in four-dimensional spacetime. "
        "Use the output from Sub-task 1 as context and debate to reach consensus."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Calculate the mass dimension of the coupling constant kappa by ensuring the interaction Lagrangian term "
        "has the correct overall mass dimension, based on outputs from Sub-tasks 1 and 2. Use self-consistency chain-of-thought."
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
        "Sub-task 4: Assess the renormalizability of the theory based on the mass dimension of kappa "
        "and standard power-counting arguments in quantum field theory, using output from Sub-task 3."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Combine the results of the mass dimension calculation and renormalizability assessment "
        "to select the correct choice from the given options, using self-consistency chain-of-thought."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
