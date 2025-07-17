async def forward_164(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the catalyst systems involved in ethylene polymerization, "
        "including the initial homogeneous organometallic catalyst and the second catalyst system intended to introduce regular branches using only ethylene, "
        "with context from the provided query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze and classify the activators mentioned, focusing on aluminum-based activators and specific activators compatible with group VIa transition metal catalysts, "
        "and their roles in the essential additional reaction step, with context from the provided query."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Analyze and classify the industrial and economic context of the catalyst systems, "
        "including the implementation of combined catalyst systems in the US and the cost implications of noble metal catalysts, with context from the provided query."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results3, log3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the correctness of each of the four statements regarding the formation of branched polymers using only ethylene and a dual catalyst system, "
        "based on the analysis of catalyst systems, activators, and industrial context from Subtasks 1, 2, and 3."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results1, results2, results3],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])

    return final_answer, logs
