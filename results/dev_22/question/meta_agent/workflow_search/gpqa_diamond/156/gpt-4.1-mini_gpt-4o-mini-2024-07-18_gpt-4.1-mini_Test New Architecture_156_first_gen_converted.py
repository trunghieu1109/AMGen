async def forward_156(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Extract and characterize essential features of the retrovirus outbreak, including viral genome type, "
        "possible molecular targets (viral nucleic acids or antibodies), and diagnostic constraints such as speed and accuracy, "
        "based on the provided query."
    )
    debate_desc_stage0 = {
        'instruction': cot_instruction_stage0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query", "stage 0: feature extraction"]
    }
    results_stage0, log_stage0 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=debate_desc_stage0,
        n_repeat=self.max_round
    )
    logs.append(log_stage0)

    cot_instruction_stage1 = (
        "Sub-task 1: Analyze and classify diagnostic approaches based on extracted features from stage 0 output: nucleic acid-based methods "
        "(DNA sequencing, cDNA sequencing, PCR variants) versus antibody-based methods (IgG detection via ELISA), and symptom-based inference."
    )
    debate_desc_stage1 = {
        'instruction': cot_instruction_stage1,
        'input': [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        'temperature': 0.5,
        'context': ["user query", "stage 0 thinking", "stage 0 answer", "stage 1: diagnostic approach classification"]
    }
    results_stage1, log_stage1 = await self.debate(
        subtask_id="stage1_subtask1",
        debate_desc=debate_desc_stage1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1)

    cot_sc_instruction_stage2 = (
        "Sub-task 1: Transform classified diagnostic approaches from stage 1 into concrete molecular diagnostic kit designs, "
        "generating variants such as PCR, nested PCR, real-time PCR, and ELISA kits, considering technical feasibility and detection targets."
    )
    results_stage2, log_stage2 = await self.sc_cot(
        subtask_id="stage2_subtask1",
        cot_agent_desc={
            'instruction': cot_sc_instruction_stage2,
            'input': [taskInfo, results_stage1['thinking'], results_stage1['answer']],
            'temperature': 0.5,
            'context': ["user query", "stage 1 thinking", "stage 1 answer", "stage 2: diagnostic kit design"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log_stage2)

    debate_instruction_stage3 = (
        "Sub-task 1: Evaluate and prioritize the generated diagnostic kit designs from stage 2 against criteria of speed, accuracy, feasibility, "
        "and suitability for retrovirus detection to select the optimal design."
    )
    debate_desc_stage3 = {
        'instruction': debate_instruction_stage3,
        'input': [taskInfo, results_stage1['thinking'], results_stage1['answer'], results_stage2['thinking'], results_stage2['answer']],
        'temperature': 0.5,
        'context': ["user query", "stage 1 thinking", "stage 1 answer", "stage 2 thinking", "stage 2 answer", "stage 3: evaluation and prioritization"]
    }
    results_stage3, log_stage3 = await self.debate(
        subtask_id="stage3_subtask1",
        debate_desc=debate_desc_stage3,
        n_repeat=self.max_round
    )
    logs.append(log_stage3)

    final_answer = await self.make_final_answer(results_stage3['thinking'], results_stage3['answer'])
    return final_answer, logs
