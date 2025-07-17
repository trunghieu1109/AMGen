async def forward_197(self, taskInfo):
    logs = []

    cot_instruction0 = (
        "Sub-task 1: Extract and summarize all given quantitative data and constants relevant to the problem, "
        "including total cobalt concentration, SCN- concentration, and cumulative stability constants for all cobalt(II) thiocyanato complexes. "
        "Provide a clear summary with values and units."
    )
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0, log0 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc0
    )
    logs.append(log0)

    cot_sc_instruction1_1 = (
        "Sub-task 1: Calculate the free Co(II) ion concentration by setting up the mass balance equation for cobalt and "
        "expressing complex concentrations in terms of free Co(II) and SCN- concentrations using the cumulative stability constants. "
        "Use the summary from stage_0.subtask_1 as context."
    )
    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_sc_instruction1_2 = (
        "Sub-task 2: Compute the concentrations of all cobalt(II) thiocyanato complexes (Co(SCN)+, Co(SCN)2, Co(SCN)3-, Co(SCN)4 2-) "
        "using the free Co(II) concentration and the given SCN- concentration. Use results from stage_1.subtask_1."
    )
    cot_sc_desc1_2 = {
        'instruction': cot_sc_instruction1_2,
        'input': [taskInfo, results1_1['thinking'], results1_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    cot_reflect_instruction2 = (
        "Sub-task 1: Derive the fraction of the blue dithiocyanato complex (Co(SCN)2) relative to the total cobalt concentration "
        "by dividing its concentration by the sum of all cobalt species concentrations. Use outputs from stage_1.subtask_2."
    )
    critic_instruction2 = (
        "Please review the fraction calculation and provide any limitations or assumptions made."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1_2['thinking'], results1_2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"
        ]
    }
    results2, log2 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 1: Convert the fraction of the dithiocyanato complex to a percentage and compare it with the given answer choices "
        "to select the closest match. Use outputs from stage_2.subtask_1."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results2['thinking'], results2['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
