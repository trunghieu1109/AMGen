async def forward_152(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Extract and summarize the given chemical information, including reactants, reagents, "
        "reaction conditions, and definitions related to Michael addition reactions from the provided query."
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

    debate_instruction_stage1 = (
        "Sub-task 1: Derive the expected major products and intermediates of the three Michael addition reactions "
        "by applying knowledge of reaction mechanisms, regioselectivity, tautomerization, and resonance stabilization, "
        "based on the summarized chemical information from Stage 0."
    )
    debate_desc_stage1 = {
        'instruction': debate_instruction_stage1,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    debate_instruction_stage2 = (
        "Sub-task 1: Evaluate and prioritize the four multiple-choice options by comparing their proposed products "
        "with the deduced products from Stage 1 to identify the best matching set of products A, B, and C."
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
