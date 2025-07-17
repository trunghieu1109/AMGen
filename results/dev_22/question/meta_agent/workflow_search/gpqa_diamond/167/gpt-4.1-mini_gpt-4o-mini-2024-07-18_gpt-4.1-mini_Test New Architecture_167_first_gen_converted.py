async def forward_167(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each of the four issues in genomics data analysis: "
        "mutually incompatible data formats, 'chr' / 'no chr' confusion, reference assembly mismatch, incorrect ID conversion. "
        "Determine their definitions, causes, and potential impact with context from the query."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Evaluate the relationships and overlaps between the four issues analyzed in Sub-task 1, "
        "including how they contribute to data incompatibility and difficult-to-spot erroneous results in genomics data analysis. "
        "Identify which issues are inherently more difficult to detect."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", results1['thinking'], results1['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Based on the outputs from Sub-task 2, select and prioritize the issues by their frequency and impact "
        "as common sources of difficult-to-spot errors in genomics data analysis, considering practical scenarios and workflows."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results2['thinking'], results2['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Compare the prioritized issues from Sub-task 3 against the provided answer choices, "
        "and determine which combination best represents the most common sources of difficult-to-spot erroneous results in genomics data analysis."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", results3['thinking'], results3['answer']]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
