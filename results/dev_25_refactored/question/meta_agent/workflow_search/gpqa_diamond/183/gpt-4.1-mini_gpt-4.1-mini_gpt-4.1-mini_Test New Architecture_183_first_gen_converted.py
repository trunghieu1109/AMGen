async def forward_183(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and interpret each individual reaction step in the given sequences, "
        "identifying the chemical transformation it performs and its effect on the benzene ring, "
        "with context from the user query."
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
        "Sub-task 2: Based on the output from Sub-task 1, integrate the individual reaction steps "
        "into coherent synthetic pathways for each choice, considering directing effects, substitution patterns, "
        "and functional group compatibility to predict substitution positions and intermediate structures."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent integrated synthetic pathway for each choice."
    )
    N = self.max_sc
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
        n_repeat=N
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate each integrated synthetic pathway against the target molecule's substitution pattern "
        "(2-tert-butyl, 1-ethoxy, 3-nitro) and reaction feasibility to identify which sequence leads to the high-yield synthesis."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the best synthetic sequence from the given choices based on the evaluation."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
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
