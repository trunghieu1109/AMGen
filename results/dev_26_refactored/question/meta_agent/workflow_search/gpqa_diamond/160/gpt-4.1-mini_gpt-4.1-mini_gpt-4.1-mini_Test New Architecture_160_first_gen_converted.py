async def forward_160(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given information from the query, including the physical setup, vacuum conditions, initial mean free path λ1, and the observed change to λ2 upon electron beam initiation."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Clarify definitions and physical meaning of mean free path λ1 and λ2, including the role of gas molecule collisions and electron scattering, and identify assumptions such as constant temperature and vacuum pressure."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for clarifying the physical meaning and assumptions of λ1 and λ2."
    )
    cot_agent_desc2 = {
        "instruction": cot_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Combine and integrate physical principles from kinetic theory, scattering theory, and electron-matter interactions to analyze how the electron beam affects the effective mean free path, and establish the theoretical relationship between λ1 and λ2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent theoretical relationship between λ1 and λ2 based on the integrated analysis."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the given multiple-choice options against the integrated analysis to select the correct conclusion about the relationship between λ2 and λ1."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the correct conclusion about the relationship between λ2 and λ1 based on the debate."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
