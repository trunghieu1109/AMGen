async def forward_195(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Explicitly extract and list all given symbolic physical parameters and constants relevant to the problem, including mass m, spring constant k, amplitude A, and speed of light c. Avoid any assumption that parameters are missing to maintain full symbolic context for subsequent reasoning."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, identify and define key dimensionless parameters characterizing the relativistic regime of the oscillator, such as epsilon = kA^2/(2mc^2), and analyze their physical meaning and domain of validity. This will guide the relativistic corrections and ensure the formulas remain physically meaningful."
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent and correct dimensionless parameters and their physical interpretation for the relativistic harmonic oscillator problem."
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

    debate_instruction3 = "Sub-task 3: Derive the relativistic equation of motion for the harmonic oscillator, explicitly incorporating relativistic momentum (p = gamma*m*v) and force relations. Avoid assuming classical potential energy forms; carefully analyze how Hooke's law translates in the relativistic context and how it affects energy conservation."
    final_decision_instruction3 = "Sub-task 3: Provide a rigorous derivation of the relativistic equation of motion for the harmonic oscillator, highlighting key assumptions and implications."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = "Sub-task 4: Using the relativistic equation of motion and energy relations from Subtask 3, derive an explicit analytic expression for the maximum speed v_max of the mass. Ensure the derivation respects relativistic constraints (v_max < c) and carefully handle the Lorentz factor without oversimplifying the potential energy term."
    final_decision_instruction4 = "Sub-task 4: Synthesize the derivation and present the explicit analytic formula for v_max consistent with relativistic dynamics."
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Map the derived analytic formula for v_max precisely to the provided candidate formulas (choices 1 to 4). Perform symbolic matching and check for equivalence or approximations rigorously. Avoid premature or groupthink-based selection."
    final_decision_instruction5 = "Sub-task 5: Identify which candidate formula matches the derived expression exactly or within valid approximations."
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

    cot_reflect_instruction6 = "Sub-task 6: Perform a final sanity check and physical consistency validation of the selected candidate formula. Verify that the formula respects physical constraints (v_max < c, positivity under square roots), domain of validity of parameters, and consistency with relativistic dynamics. Reflect on the entire reasoning process to ensure no previous errors are repeated."
    critic_instruction6 = "Please review and provide limitations or potential issues of the selected candidate formula and the overall solution."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'critic_instruction': critic_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 2", "answer of subtask 2"]
    }
    results6, log6 = await self.reflexion(
        subtask_id="subtask_6",
        reflect_desc=cot_reflect_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
