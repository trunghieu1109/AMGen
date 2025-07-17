async def forward_161(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Extract and summarize the given metric, domain, and problem parameters, including the radius and the nature of the pseudosphere, with context from taskInfo"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Analyze the geometric and analytic properties of the metric, including the conformal factor, domain restrictions, and implications for curvature and boundary behavior, based on output from Sub-task 1"
    results2, log2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc={
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        },
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Compute the area element induced by the metric and set up the integral expression for the total area of the pseudosphere of radius 2, based on output from stage_0.subtask_2"
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results3, log3 = await self.cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    logs.append(log3)

    cot_reflect_instruction4 = "Sub-task 4: Evaluate the area integral explicitly or by known formulas for the pseudosphere, and interpret the result in the context of the given answer choices, based on outputs from stage_1.subtask_3"
    critic_instruction4 = "Please review the evaluation and provide its limitations."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="stage_2.subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Based on the output of Sub-task 4, convert the evaluated area into the final answer by matching it with one of the provided choices and provide a clear justification"
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="stage_2.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
