async def forward_170(self, taskInfo):
    logs = []

    cot_instruction0 = (
        "Sub-task 1: Extract and summarize the chemical nature and substituent characteristics of substances 1-6, "
        "including their electronic effects and directing influence on electrophilic aromatic substitution."
    )
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results0, log0 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=cot_agent_desc0,
        n_repeat=self.max_round
    )
    logs.append(log0)

    cot_sc_instruction1_1 = (
        "Sub-task 1: Analyze how each substituent's electronic and steric properties affect the regioselectivity of bromination, "
        "focusing on the relative yields of para versus ortho isomers, based on the summary from stage 0."
    )
    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage1_subtask1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_reflect_instruction1_2 = (
        "Sub-task 2: Estimate or rationalize the expected weight fraction of the para-isomer for each substance "
        "based on substituent effects and steric hindrance, using outputs from stage 0 and stage 1 subtask 1."
    )
    cot_reflect_desc1_2 = {
        'instruction': cot_reflect_instruction1_2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1_1['thinking'], results1_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results1_2, log1_2 = await self.reflexion(
        subtask_id="stage1_subtask2",
        reflect_desc=cot_reflect_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Compare and prioritize the six substances by increasing weight fraction of the para-bromo isomer yield, "
        "integrating insights from previous analyses (stage 1 subtasks 1 and 2)."
    )
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'context': ["user query", results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    cot_sc_instruction2_2 = (
        "Sub-task 2: Match the prioritized order of substances by para-isomer weight fraction with the given multiple-choice options "
        "and select the correct answer, based on the output from stage 2 subtask 1."
    )
    cot_sc_desc2_2 = {
        'instruction': cot_sc_instruction2_2,
        'input': [taskInfo, results2_1['thinking'], results2_1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage2_subtask1", "answer of stage2_subtask1"]
    }
    results2_2, log2_2 = await self.sc_cot(
        subtask_id="stage2_subtask2",
        cot_agent_desc=cot_sc_desc2_2,
        n_repeat=self.max_sc
    )
    logs.append(log2_2)

    final_answer = await self.make_final_answer(results2_2['thinking'], results2_2['answer'])
    return final_answer, logs
