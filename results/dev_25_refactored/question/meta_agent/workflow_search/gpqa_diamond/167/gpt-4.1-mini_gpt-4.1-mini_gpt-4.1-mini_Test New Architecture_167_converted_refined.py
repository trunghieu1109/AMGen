async def forward_167(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the four specific issues related to genomics data analysis and the given answer choices, "
        "clarifying their definitions and contexts. Explicitly identify the nature of each issue and the entities involved to provide a clear foundation for subsequent analysis."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Classify each of the four issues by the type of errors they cause, explicitly distinguishing between immediate, obvious failures and subtle, difficult-to-spot erroneous results. "
        "Include an assumption check phase to critically evaluate and validate assumptions about error detectability, especially for mutually incompatible data formats, to avoid oversimplified conclusions."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent classification for each issue regarding error detectability and subtlety."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Integrate domain knowledge and empirical evidence about the prevalence and impact of each issue in typical genomics data analysis pipelines. "
        "Assess how common each issue is as a source of subtle errors, refine the classification from subtask 2 accordingly, and explicitly address the ambiguity in 'most common' and 'difficult-to-spot' criteria ensuring consistent definitions."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of the provided classifications and prevalence assessments, highlighting any assumptions or gaps."
    )
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="stage_1.subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Stage 2 Sub-task 1: Evaluate and select the combination of issues from the given answer choices that best represents the most common sources of difficult-to-spot erroneous results in genomics data analysis. "
        "Leverage the refined classifications and domain knowledge from stage_1 subtasks, incorporate a consensus-building process to resolve any remaining ambiguities or conflicts, and explicitly justify the final choice with clear reasoning."
    )
    final_decision_instruction4 = (
        "Stage 2 Sub-task 1: Select and justify the best answer choice representing the most common sources of difficult-to-spot erroneous results in genomics data analysis."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
