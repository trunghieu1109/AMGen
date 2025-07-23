async def forward_167(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and categorize the four given issues and the four answer choices, "
        "clearly defining their attributes and relevance to genomics data analysis."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 1: Analyze and characterize the relationships, interactions, and potential error impacts "
        "of the four issues, focusing on their roles in causing difficult-to-spot erroneous results in genomics workflows."
    )
    final_decision_instruction2 = (
        "Sub-task 1: Synthesize and choose the most consistent analysis of the issues' relationships and impacts."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 1: Derive a transformed output by evaluating the frequency and impact of each issue "
        "based on domain knowledge and validate which combinations of issues best represent the most common sources of errors."
    )
    final_decision_instruction3 = (
        "Sub-task 1: Synthesize and choose the most consistent evaluation of issue frequency and impact."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 1: Select the best candidate answer choice from the given options that most accurately reflects "
        "the most common sources of difficult-to-spot erroneous results in genomics data analysis, based on the validated analysis."
    )
    final_decision_instruction4 = (
        "Sub-task 1: Choose the best answer choice reflecting the most common error sources in genomics data analysis."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_4.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
