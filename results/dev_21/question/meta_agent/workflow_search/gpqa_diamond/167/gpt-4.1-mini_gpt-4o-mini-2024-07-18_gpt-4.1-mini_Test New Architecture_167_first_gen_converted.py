async def forward_167(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Extract and summarize detailed information about each of the four issues "
        "(mutually incompatible data formats, 'chr' / 'no chr' confusion, reference assembly mismatch, incorrect ID conversion) "
        "relevant to genomics data analysis."
    )
    cot_agent_desc_stage0 = {
        'instruction': cot_instruction_stage0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0, log_stage0 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0
    )
    logs.append(log_stage0)

    cot_sc_instruction_stage1 = (
        "Sub-task 1: Analyze and classify each issue based on how commonly it occurs and how difficult it is to detect "
        "as a source of erroneous results in genomics data analysis, using the detailed summaries from Stage 0."
    )
    cot_sc_desc_stage1 = {
        'instruction': cot_sc_instruction_stage1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1, log_stage1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_stage1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Evaluate and prioritize the issues identified in Stage 1 to determine which combination best represents "
        "the most common and difficult-to-spot error sources, and select the corresponding answer choice."
    )
    debate_desc_stage2 = {
        'instruction': debate_instruction_stage2,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'input': [taskInfo, results_stage1['thinking'], results_stage1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2, log_stage2 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2)

    final_answer = await self.make_final_answer(results_stage2['thinking'], results_stage2['answer'])

    return final_answer, logs
