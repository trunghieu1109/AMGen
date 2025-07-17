async def forward_153(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize the key spectral features from the mass spectrum, IR spectrum, and 1H NMR data "
        "to characterize the unknown compound, based on the provided spectral data and candidate structures."
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

    cot_sc_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the extracted spectral features from Stage 0 to identify functional groups, isotopic patterns, "
        "and aromatic substitution patterns, and classify the compound's structural characteristics."
    )
    cot_sc_desc_stage1_sub1 = {
        'instruction': cot_sc_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage1_subtask1",
        cot_agent_desc=cot_sc_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Assess the impact of the identified spectral features on possible structural isomers and evaluate the consistency "
        "of each candidate structure with the spectral data, based on outputs from Stage 0 and Stage 1 Subtask 1."
    )
    cot_reflect_desc_stage1_sub2 = {
        'instruction': cot_reflect_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage1_subtask1", "answer of stage1_subtask1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage1_subtask2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Select the most reasonable structural suggestion for the unidentified compound by comparing the candidate structures "
        "against the analyzed spectral data and criteria derived from previous stages."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'context': ["user query", results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(results_stage2_sub1['thinking'], results_stage2_sub1['answer'])
    return final_answer, logs
