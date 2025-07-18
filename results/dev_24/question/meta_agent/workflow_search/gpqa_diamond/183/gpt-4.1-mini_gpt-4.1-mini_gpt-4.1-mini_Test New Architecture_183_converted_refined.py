async def forward_183(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Perform a detailed mechanistic feasibility check for each reaction step in the given sequences starting from benzene. "
        "Explicitly verify whether each substituent introduction (tert-butyl, ethoxy, nitro) and subsequent transformations are chemically plausible under the stated conditions. "
        "Identify any steps that rely on incorrect assumptions and propose alternative feasible synthetic tactics if necessary. "
        "This subtask addresses the critical failure in previous attempts where mechanistic feasibility was not questioned, preventing propagation of unrealistic synthetic steps."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage0_subtask1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, analyze the directing effects and regioselectivity of each feasible reaction step and map the positional outcomes of substituents on the benzene ring throughout the sequences. "
        "Predict intermediate substitution patterns and verify alignment with the target substitution pattern (2-tert-butyl, 1-ethoxy, 3-nitro). "
        "This separation ensures regioselectivity analysis is grounded on chemically realistic transformations."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for regioselectivity analysis. "
        "Given all the above thinking and answers, find the most consistent and correct positional outcomes for the substitution pattern."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage0_subtask2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate and compare the four given sequences based on mechanistic feasibility (from stage0_subtask1), regioselectivity (from stage0_subtask2), and overall synthetic practicality. "
        "Identify which sequences are likely to lead to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene. "
        "Incorporate critical assessment of reaction order, compatibility of conditions, and potential synthetic bottlenecks, explicitly avoiding assumptions of impossible steps."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a reasoned evaluation and ranking of the sequences based on the above criteria."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage0_subtask2", "answer of stage0_subtask2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage1_subtask3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Identify potential side reactions, yield-limiting steps, regioselectivity challenges, and any mechanistic pitfalls in each sequence. "
        "Assess their impact on overall synthetic efficiency, product purity, and scalability. "
        "Integrate insights from mechanistic feasibility, regioselectivity, and sequence evaluation to provide a comprehensive risk and yield analysis."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide a detailed risk and yield analysis for each sequence, highlighting critical challenges and their implications."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of stage0_subtask1", "answer of stage0_subtask1", "thinking of stage0_subtask2", "answer of stage0_subtask2", "thinking of stage1_subtask3", "answer of stage1_subtask3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage1_subtask4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Select and justify the optimal sequence of reactions from the given options that leads to the high-yield synthesis of 2-(tert-butyl)-1-ethoxy-3-nitrobenzene starting from benzene. "
        "Justification must be grounded in mechanistic feasibility, regioselectivity, synthetic practicality, and yield considerations established in previous subtasks. "
        "Explicitly reference the avoidance of previously identified flawed assumptions and demonstrate a robust synthetic rationale."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final recommended sequence with detailed justification."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of stage1_subtask3", "answer of stage1_subtask3", "thinking of stage1_subtask4", "answer of stage1_subtask4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage1_subtask5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
