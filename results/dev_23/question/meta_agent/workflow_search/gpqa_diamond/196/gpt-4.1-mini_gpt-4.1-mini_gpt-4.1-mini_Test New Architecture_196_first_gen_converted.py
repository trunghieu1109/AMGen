async def forward_196(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize all given quantitative and qualitative data from IR, NMR, and reaction conditions "
        "to classify functional groups and structural features of compound X, based on the provided query."
    )
    cot_agent_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.cot(
        subtask_id="stage0_subtask1",
        cot_agent_desc=cot_agent_desc_stage0_sub1
    )
    logs.append(log_stage0_sub1)

    cot_sc_instruction_stage1_sub1 = (
        "Sub-task 1: Interpret the spectral data and reaction conditions to deduce the structure of compound X "
        "and predict the chemical transformation caused by red phosphorus and HI, based on the output from Stage 0 Sub-task 1."
    )
    cot_sc_desc_stage1_sub1 = {
        'instruction': cot_sc_instruction_stage1_sub1,
        'input': [taskInfo, results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', '')],
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
        "Sub-task 2: Map the predicted structural changes and spectral interpretations to the possible final products "
        "provided in the choices, based on outputs from Stage 1 Sub-task 1 and Stage 0 Sub-task 1."
    )
    cot_reflect_desc_stage1_sub2 = {
        'instruction': cot_reflect_instruction_stage1_sub2,
        'input': [
            taskInfo,
            results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', ''),
            results_stage1_sub1.get('thinking', ''), results_stage1_sub1.get('answer', '')
        ],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of stage0_subtask1", "answer of stage0_subtask1",
            "thinking of stage1_subtask1", "answer of stage1_subtask1"
        ]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage1_subtask2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate and prioritize the candidate products by comparing their structural features "
        "with the predicted product from the reaction and spectral analysis to identify the final product, "
        "based on outputs from Stage 1 Sub-tasks 1 and 2."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'context': [
            "user query",
            results_stage1_sub1.get('thinking', ''), results_stage1_sub1.get('answer', ''),
            results_stage1_sub2.get('thinking', ''), results_stage1_sub2.get('answer', '')
        ],
        'input': [
            taskInfo,
            results_stage1_sub1.get('thinking', ''), results_stage1_sub1.get('answer', ''),
            results_stage1_sub2.get('thinking', ''), results_stage1_sub2.get('answer', '')
        ],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage2_subtask1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(
        results_stage2_sub1.get('thinking', ''),
        results_stage2_sub1.get('answer', '')
    )

    return final_answer, logs
