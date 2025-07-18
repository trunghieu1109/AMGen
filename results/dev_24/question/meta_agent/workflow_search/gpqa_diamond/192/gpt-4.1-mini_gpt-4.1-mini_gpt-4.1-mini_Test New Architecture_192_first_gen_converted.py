async def forward_192(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Establish the mathematical relationship between parallax (plx) and distance (r), confirming that plx ∝ 1/r and expressing plx as a function of r." 
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, rewrite the given star count dependence N(plx) ∝ 1/plx^5 in terms of distance r using the relationship from subtask_1." 
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent answer for rewriting N(plx) in terms of r. Given all the above thinking and answers, find the most consistent and correct solution for the problem."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Apply the change of variables formula to convert the number of stars per unit parallax interval to the number of stars per unit distance interval, including calculation of the Jacobian |d(plx)/d(r)|." 
    final_decision_instruction3 = "Sub-task 3: Provide the transformed expression for the number of stars per unit distance interval with detailed reasoning and final formula."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Analyze the transformed expression for the number of stars per unit distance interval, simplify it, and determine its dependence on r." 
    final_decision_instruction4 = "Sub-task 4: Provide the simplified dependence of the number of stars per unit distance interval on r with clear explanation." 
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Compare the derived dependence with the given choices (~ r^2, ~ r^3, ~ r^4, ~ r^5) and select the correct option." 
    final_decision_instruction5 = "Sub-task 5: Select the correct choice that matches the derived dependence of the number of stars per unit distance interval on r." 
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context_desc': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
