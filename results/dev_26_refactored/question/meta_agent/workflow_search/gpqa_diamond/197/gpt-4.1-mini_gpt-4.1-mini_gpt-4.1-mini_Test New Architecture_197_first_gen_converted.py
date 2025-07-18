async def forward_197(self, taskInfo):
    logs = []

    cot_sc_instruction0_1 = (
        "Sub-task 1: Extract, summarize, and confirm all given information and assumptions, "
        "including concentrations, stability constants, species identities, and assumptions about constant SCN- concentration."
    )
    cot_sc_desc0_1 = {
        'instruction': cot_sc_instruction0_1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent extraction and confirmation of given data and assumptions.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    cot_instruction1_1 = (
        "Sub-task 1: Formulate the equilibrium expressions for all cobalt(II) thiocyanato complexes "
        "using the cumulative stability constants and free ion concentrations."
    )
    cot_agent_desc1_1 = {
        'instruction': cot_instruction1_1,
        'input': [taskInfo, results0_1['answer']],
        'temperature': 0.0,
        'context': ["user query", "extracted data and assumptions"]
    }
    results1_1, log1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1_1
    )
    logs.append(log1_1)

    cot_reflect_instruction1_2 = (
        "Sub-task 2: Set up the mass balance equation for total cobalt concentration as the sum of free Co(II) "
        "and all complex species concentrations expressed in terms of free Co(II) and SCN- concentrations."
    )
    critic_instruction1_2 = (
        "Please review and provide the limitations of the mass balance equation setup and assumptions made."
    )
    cot_reflect_desc1_2 = {
        'instruction': cot_reflect_instruction1_2,
        'critic_instruction': critic_instruction1_2,
        'input': [taskInfo, results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "extracted data", "equilibrium expressions thinking", "equilibrium expressions answer"]
    }
    results1_2, log1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Solve the mass balance equation to find the free Co(II) concentration in solution, "
        "using the equilibrium expressions and mass balance from previous subtasks."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Determine the most accurate free Co(II) concentration solution from the debate."
    )
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'final_decision_instruction': final_decision_instruction2_1,
        'input': [taskInfo, results1_1['answer'], results1_2['answer']],
        'context_desc': ["user query", "equilibrium expressions", "mass balance equation setup"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    debate_instruction2_2 = (
        "Sub-task 2: Calculate the concentration of the dithiocyanato complex Co(SCN)2 using the free Co(II) concentration "
        "and equilibrium expressions from previous subtasks."
    )
    final_decision_instruction2_2 = (
        "Sub-task 2: Determine the most accurate concentration of the dithiocyanato complex from the debate."
    )
    debate_desc2_2 = {
        'instruction': debate_instruction2_2,
        'final_decision_instruction': final_decision_instruction2_2,
        'input': [taskInfo, results1_1['answer'], results2_1['answer']],
        'context_desc': ["user query", "equilibrium expressions", "free Co(II) concentration"],
        'temperature': 0.5
    }
    results2_2, log2_2 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc2_2,
        n_repeat=self.max_round
    )
    logs.append(log2_2)

    debate_instruction3_1 = (
        "Sub-task 1: Calculate the percentage of the dithiocyanato cobalt(II) complex relative to total cobalt concentration "
        "and select the closest matching answer choice from the provided options."
    )
    final_decision_instruction3_1 = (
        "Sub-task 1: Provide the final percentage and select the closest answer choice."
    )
    debate_desc3_1 = {
        'instruction': debate_instruction3_1,
        'final_decision_instruction': final_decision_instruction3_1,
        'input': [taskInfo, results2_2['answer']],
        'context_desc': ["user query", "dithiocyanato complex concentration"],
        'temperature': 0.5
    }
    results3_1, log3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3_1,
        n_repeat=self.max_round
    )
    logs.append(log3_1)

    final_answer = await self.make_final_answer(results3_1['thinking'], results3_1['answer'])

    return final_answer, logs
