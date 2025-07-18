async def forward_191(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and clearly define all given parameters, vectors, and geometric relationships from the problem statement, "
        "including the positions of the conductor center, cavity center, charge, and point P, as well as the angles and distances involved."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent definitions and extractions of parameters and vectors.",
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
        "Sub-task 2: Based on the output from Sub-task 1, apply coordinate transformations or vector decompositions to express the position vectors and distances (l, s, L) "
        "and the angle theta in a consistent coordinate system to facilitate further analysis."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent coordinate system and vector expressions.",
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

    cot_reflect_instruction3 = (
        "Sub-task 3: Combine electrostatic principles with the geometric setup to analyze the effect of the charge +q inside the cavity on the conductor, "
        "including induced charges and the resulting field outside the conductor, using superposition and boundary conditions."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions regarding the induced charges and field outside the conductor, "
        "and clarify the physical correctness of the analysis."
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
        "Sub-task 4: Evaluate the candidate electric field expressions given in the choices against the physical and geometric analysis "
        "to identify which formula correctly represents the magnitude of the electric field at point P outside the conductor."
    )
    final_decision_instruction4 = "Sub-task 4: Select the correct electric field expression based on the analysis and physical principles."
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
