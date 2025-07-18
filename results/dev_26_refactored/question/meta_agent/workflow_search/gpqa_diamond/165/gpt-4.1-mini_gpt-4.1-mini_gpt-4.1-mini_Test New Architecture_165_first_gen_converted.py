async def forward_165(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the key theoretical components from the given Lagrangian and vacuum expectation values, "
        "including particle content, symmetry breaking pattern, and relevant mass parameters."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, analyze and integrate the contributions of various particles to the radiative corrections "
        "of the pseudo-Goldstone boson mass, focusing on the structure of the loop corrections and the role of each mass term and coefficient."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the analysis of radiative corrections contributions."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the candidate formulae for the pseudo-Goldstone boson mass squared against physical and theoretical criteria "
        "such as sign conventions for fermionic and bosonic contributions, dimensional consistency, and inclusion of relevant particle masses."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a reasoned evaluation and ranking of the candidate formulae based on the criteria."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Select the most appropriate approximation formula for the pseudo-Goldstone boson mass squared from the given choices "
        "based on the evaluation in Sub-task 3."
    )
    cot_agent_desc4 = {
        "instruction": cot_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
