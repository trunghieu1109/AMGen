async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given quantitative data and physical constraints from the problem statement, "
        "including initial mass, fragment mass ratio, rest-mass energy loss, and conservation laws."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Derive the rest masses of the two fragments based on the given mass ratio and total rest mass after fission, "
        "and determine the total kinetic energy available from the 1% mass loss."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct solutions for the fragment masses and total kinetic energy."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Using conservation of momentum and energy, calculate the velocities and momenta of both fragments, "
        "ensuring relativistic relations are applied correctly."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent and correct solutions for velocities and momenta."
    )
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

    cot_sc_instruction4 = (
        "Sub-task 4: Compute the kinetic energy of the heavier fragment using both the relativistic formula "
        "and the classical (non-relativistic) approximation, based on velocities and momenta from Sub-task 3."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent and correct kinetic energy values for the heavier fragment."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Calculate the difference between the relativistic kinetic energy and the classical kinetic energy of the heavier fragment, "
        "and interpret the result in the context of the given multiple-choice answers."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final answer choice that best matches the calculated difference in kinetic energies."
    )
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
