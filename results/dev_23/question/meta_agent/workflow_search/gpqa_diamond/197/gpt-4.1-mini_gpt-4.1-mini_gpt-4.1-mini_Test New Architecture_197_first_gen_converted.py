async def forward_197(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize all given information including concentrations, stability constants, "
        "and identify the target species (blue dithiocyanato complex) with context from the query."
    )
    cot_agent_desc_stage0_sub1 = {
        'instruction': cot_instruction_stage0_sub1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1
    )
    logs.append(log_stage0_sub1)

    debate_instruction_stage1_sub1 = (
        "Sub-task 1: Calculate the equilibrium concentrations of all cobalt(II) thiocyanato complexes "
        "(Co(SCN)n, n=0 to 4) using the cumulative stability constants and given total concentrations, "
        "based on the extracted information from stage_0.subtask_1."
    )
    debate_desc_stage1_sub1 = {
        'instruction': debate_instruction_stage1_sub1,
        'context': ["user query", results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Determine the free Co(II) concentration and verify the assumption that free SCN- concentration "
        "remains approximately 0.1 M or adjust calculations accordingly, based on the extracted information from stage_0.subtask_1."
    )
    cot_reflect_critic_instruction_stage1_sub2 = (
        "Please review the assumption about free SCN- concentration and provide its limitations."
    )
    cot_reflect_desc_stage1_sub2 = {
        'instruction': cot_reflect_instruction_stage1_sub2,
        'input': [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            results_stage0_sub1['thinking'],
            results_stage0_sub1['answer']
        ]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    cot_instruction_stage1_sub3 = (
        "Sub-task 3: Integrate results from equilibrium calculations (stage_1.subtask_1) and free ion concentration analysis "
        "(stage_1.subtask_2) to obtain consistent species distribution."
    )
    cot_agent_desc_stage1_sub3 = {
        'instruction': cot_instruction_stage1_sub3,
        'input': [
            taskInfo,
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ],
        'temperature': 0.0,
        'context': [
            "user query",
            results_stage1_sub1['thinking'], results_stage1_sub1['answer'],
            results_stage1_sub2['thinking'], results_stage1_sub2['answer']
        ]
    }
    results_stage1_sub3, log_stage1_sub3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc_stage1_sub3
    )
    logs.append(log_stage1_sub3)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Calculate the percentage of the blue dithiocyanato complex (Co(SCN)2) relative to the total cobalt concentration "
        "using the equilibrium concentrations obtained from stage_1.subtask_3."
    )
    debate_desc_stage2_sub1 = {
        'instruction': debate_instruction_stage2_sub1,
        'context': ["user query", results_stage1_sub3['thinking'], results_stage1_sub3['answer']],
        'input': [taskInfo, results_stage1_sub3['thinking'], results_stage1_sub3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    debate_instruction_stage2_sub2 = (
        "Sub-task 2: Compare the calculated percentage of the blue dithiocyanato complex with the provided choices "
        "and select the closest matching answer."
    )
    debate_desc_stage2_sub2 = {
        'instruction': debate_instruction_stage2_sub2,
        'context': ["user query", results_stage2_sub1['thinking'], results_stage2_sub1['answer']],
        'input': [taskInfo, results_stage2_sub1['thinking'], results_stage2_sub1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results_stage2_sub2, log_stage2_sub2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc_stage2_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub2)

    final_answer = await self.make_final_answer(results_stage2_sub2['thinking'], results_stage2_sub2['answer'])
    return final_answer, logs
