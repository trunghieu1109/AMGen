async def forward_157(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given biological and genetic information from the query, including the roles of phosphorylation, dimerization, mutation X and Y characteristics, and the provided phenotype options."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 1: Integrate and analyze the summarized information to understand the molecular mechanism of the transcription factor activation and the impact of mutations X and Y, focusing on the dominant-negative effect of mutation Y in the dimerization domain."
    )
    final_decision_instruction2 = (
        "Sub-task 1: Synthesize and choose the most consistent understanding of the molecular mechanism and dominant-negative effect of mutation Y."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 1: Evaluate each molecular phenotype option against the integrated analysis to identify which phenotype best explains the dominant-negative effect of mutation Y."
    )
    final_decision_instruction3 = (
        "Sub-task 1: Select the molecular phenotype option that most likely represents the dominant-negative effect of mutation Y."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
