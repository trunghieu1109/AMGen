async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and interpret each individual reaction step in the given sequences, "
        "identifying the chemical transformation it performs and its effect on the benzene ring. "
        "Explicitly note the expected substituent introduced or modified, and the typical directing effects of the reagents used. "
        "Avoid assumptions about substituent positions at this stage; focus on reaction type and chemical feasibility."
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
        "Sub-task 2: Based on the output from Sub-task 1, create and maintain a detailed Position Tracker that assigns absolute numbering to the benzene ring carbons starting from benzene and updates the substituent map after each reaction step in every sequence. "
        "Explicitly record the position of each substituent after every step, ensuring no positional ambiguity or conflation occurs. "
        "The output should be a clear table or coordinate list showing substituent identity and position after each step."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and accurate Position Tracker for all sequences."
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

    cot_instruction3 = (
        "Sub-task 3: Integrate the individual reaction steps into a coherent synthetic pathway for each choice by combining the chemical transformations (from Sub-task 1) with the Position Tracker data (from Sub-task 2). "
        "Perform a rigorous chemical feasibility analysis for each step order, considering directing effects, catalyst compatibility, and functional group tolerance. "
        "Explicitly identify and reject sequences containing chemically implausible or impractical steps. "
        "Provide a detailed rationale for the predicted substitution pattern and intermediate structures after each step."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate each integrated synthetic pathway against the target molecule's substitution pattern and overall reaction feasibility. "
        "Use a strict evaluation rubric that cross-checks regioselectivity, synthetic practicality, and yield considerations. "
        "Employ a Debate collaboration pattern involving domain-expert style agents to challenge assumptions, identify confirmation bias, and ensure the final selected sequence is both chemically sound and synthetically efficient. "
        "Provide a clear justification for the best choice and explain why other sequences fail."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best synthetic sequence for the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene."
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
