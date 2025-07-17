async def forward_168(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize the defining features of the original and variant decay processes, "
        "including particle types, emitted particles, and known spectral properties, with context from taskInfo"
    )
    cot_agent_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage0_sub1)

    cot_instruction_stage1_sub1 = (
        "Sub-task 1: Assess the impact of replacing two V particles with one massless M particle on the kinematics "
        "and energy distribution of the emitted E particles, based on output from stage0_subtask1"
    )
    cot_agent_desc_stage1_sub1 = {
        'instruction': cot_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage1_subtask1",
        debate_desc=cot_agent_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    cot_sc_instruction_stage1_sub2 = (
        "Sub-task 2: Transform the original energy spectrum model to incorporate the changes due to the variant decay "
        "and validate the expected changes in spectrum shape and endpoint, based on outputs from stage0_subtask1 and stage1_subtask1"
    )
    cot_sc_desc_stage1_sub2 = {
        'instruction': cot_sc_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.sc_cot(
        subtask_id="stage1_subtask2",
        cot_agent_desc=cot_sc_desc_stage1_sub2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub2)

    cot_sc_instruction_stage2_sub1 = (
        "Sub-task 1: Analyze and classify the resulting energy spectrum of the E particles in the variant decay, "
        "comparing continuity, shape, and endpoint to the original spectrum, based on outputs from stage1_subtask1 and stage1_subtask2"
    )
    cot_sc_desc_stage2_sub1 = {
        'instruction': cot_sc_instruction_stage2_sub1,
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage1_subtask1", "answer of stage1_subtask1", "thinking of stage1_subtask2", "answer of stage1_subtask2"]
    }
    results_stage2_sub1, log_stage2_sub1 = await self.sc_cot(
        subtask_id="stage2_subtask1",
        cot_agent_desc=cot_sc_desc_stage2_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
