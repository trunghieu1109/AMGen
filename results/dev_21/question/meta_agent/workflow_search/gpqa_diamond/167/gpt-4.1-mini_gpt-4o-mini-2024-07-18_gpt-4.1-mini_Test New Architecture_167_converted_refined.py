async def forward_167(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize detailed information about each of the four issues "
        "(mutually incompatible data formats, 'chr' / 'no chr' confusion, reference assembly mismatch, incorrect ID conversion) "
        "relevant to genomics data analysis. Include typical manifestations, contexts where they arise, "
        "and their potential impact on data analysis results, providing a solid factual basis for subsequent evaluation."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Critically analyze and classify each issue along two explicit dimensions: "
        "(1) How commonly does the issue occur in genomics data analysis? "
        "(2) How difficult is it to detect the erroneous results caused by this issue (silent/subtle errors vs obvious failures)? "
        "Provide clear justification for each classification, explicitly challenging assumptions that all issues are equally difficult to detect. "
        "Use the detailed summaries from Sub-task 1 as context."
    )
    critic_instruction2 = (
        "Please review the classifications and justifications for each issue, highlighting any limitations or biases."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate and prioritize the issues based on the refined classifications from Stage 1 Sub-task 2, "
        "focusing on identifying which combination of issues best represents the most common and difficult-to-spot sources of erroneous results in genomics data analysis. "
        "Debate and justify the inclusion or exclusion of each issue, explicitly referencing detectability and frequency criteria established earlier. "
        "Prevent groupthink and ensure a well-reasoned final selection aligned with the question's emphasis."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
