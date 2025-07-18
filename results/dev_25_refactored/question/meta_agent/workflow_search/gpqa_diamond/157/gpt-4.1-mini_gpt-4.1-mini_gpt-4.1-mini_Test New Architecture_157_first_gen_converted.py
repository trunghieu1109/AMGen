async def forward_157(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the key biological information and mutation details from the query, "
        "including the transcription factor activation mechanism, mutation X and Y characteristics, and the provided phenotype options."
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

    debate_instruction2 = (
        "Sub-task 2: Integrate and analyze the summarized information from Sub-task 1 to understand the molecular mechanism "
        "of the dominant-negative mutation Y in the dimerization domain and its likely impact on protein dimerization, stability, and function."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent explanation for the dominant-negative effect of mutation Y."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the four provided molecular phenotype options against the integrated analysis from Sub-task 2 "
        "to identify which phenotype best explains the dominant-negative effect of mutation Y."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the molecular phenotype option that most likely corresponds to the dominant-negative mutation Y."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
