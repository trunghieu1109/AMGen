async def forward_167(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify each of the four issues (mutually incompatible data formats, 'chr' / 'no chr' confusion, "
        "reference assembly mismatch, incorrect ID conversion) in terms of their characteristics, typical occurrence, and potential to cause difficult-to-spot errors in genomics data analysis. "
        "Provide detailed reasoning for each issue separately with context from the user query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Evaluate and prioritize the issues based on their frequency and subtlety in causing erroneous results, "
        "considering the analysis from Sub-task 1. Debate the relative importance and difficulty of spotting errors caused by each issue. "
        "Provide arguments for and against each issue's impact and frequency."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize the debate results and choose the most common sources of difficult-to-spot errors among the four issues, "
        "based on the debate outputs and analysis from Sub-task 1."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Compare the prioritized issues identified in Sub-task 2 against the given answer choices. "
        "Determine which combination of issues best matches the identified most common sources of difficult-to-spot errors. "
        "Provide a clear rationale for selecting the best matching choice."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct answer choice that matches the prioritized issues from Sub-task 2."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
