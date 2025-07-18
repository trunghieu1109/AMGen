async def forward_195(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given physical parameters and constants relevant to the relativistic harmonic oscillator, "
        "including mass m, spring constant k, amplitude A, and speed of light c, from the provided query."
    )
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

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the extracted parameters from Sub-task 1, compute the relativistic energy expressions "
        "and identify key dimensionless parameters such as k*A^2/(2*m*c^2) that characterize the system's relativistic regime. "
        "Consider the physical meaning and constraints of these parameters."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct relativistic energy expressions and dimensionless parameters "
        "given the outputs of Sub-task 1."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Combine the computed energy expressions and physical parameters from Sub-task 2 to formulate an expression "
        "for the velocity as a function of position and system parameters, incorporating relativistic corrections. "
        "Explain the derivation steps clearly."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and select the most consistent velocity-position relation with relativistic corrections "
        "based on the previous outputs."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Derive the explicit formula for the maximum speed v_max of the mass by applying relativistic energy conservation "
        "and harmonic oscillator constraints, simplifying the expression to a form comparable to the candidate formulas."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and select the most consistent and simplified formula for v_max based on previous derivations."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate the four candidate formulas for maximum speed v_max against the derived expression and physical constraints "
        "such as v_max < c and positivity of terms. Identify the correct formula and justify the choice."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final decision on the correct formula for v_max based on the debate and evaluation."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
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
