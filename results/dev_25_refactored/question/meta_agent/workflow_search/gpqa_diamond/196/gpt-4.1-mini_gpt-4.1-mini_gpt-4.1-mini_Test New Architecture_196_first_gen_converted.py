async def forward_196(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Interpret the given IR and 1H NMR spectral data to identify key functional groups and structural features of compound X before reaction."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent interpretation of the spectral data."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the chemical reaction conditions (red phosphorus and HI) to determine the likely chemical transformation(s) occurring to compound X."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent chemical transformation based on reaction conditions."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Integrate spectral interpretation and reaction mechanism insights to deduce the structure of compound X and predict the structure of the final product after reaction."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions of integrating spectral data and reaction analysis to deduce compound structure and final product."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the four given candidate products against the predicted final product structure and select the correct final product based on conformity with spectral and reaction data."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the correct final product from the given choices based on all analyses."
    )
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

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
