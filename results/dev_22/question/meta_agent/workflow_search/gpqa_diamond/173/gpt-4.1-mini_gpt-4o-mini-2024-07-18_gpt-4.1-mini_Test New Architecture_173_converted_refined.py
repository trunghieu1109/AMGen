async def forward_173(self, taskInfo):
    logs = []

    cot_sc_instruction1 = "Sub-task 1: Extract and quantify all given physical parameters and relationships precisely: initial nucleus rest mass M and rest energy (300 GeV), fragment mass ratio (1:2), total fragment rest mass fraction (0.99 M), and initial conditions (nucleus at rest). Express fragment rest masses numerically in terms of M and calculate their rest energies. Provide exact symbolic and numerical expressions to support subsequent calculations."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze and rigorously formulate the physical constraints and conservation laws applicable to the fission event: conservation of total energy (including rest mass and kinetic energy), conservation of momentum (initial momentum zero), and the relationship between fragment momenta and velocities. Derive explicit equations relating the common momentum p of the fragments to their rest masses and total energy. Produce exact formulas and prepare for numerical solving."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
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

    cot_reflect_instruction3 = "Sub-task 3: Numerically solve the relativistic energy-momentum equations derived in Subtask 2 to find the common momentum p of the fragments with at least three significant digits precision. Then compute the relativistic kinetic energy T1 of the heavier fragment using the exact relativistic formula T = (gamma - 1) m c^2, where gamma is the Lorentz factor calculated from the fragment's velocity derived from p. Explicitly show all numerical steps and intermediate values."
    critic_instruction3 = "Please review the numerical solution and calculations for limitations or errors."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_reflect_instruction4 = "Sub-task 4: Using the same momentum p found in Subtask 3, compute the classical (non-relativistic) kinetic energy T1_class of the heavier fragment using the formula T = p^2/(2m). Calculate and present numerical values with at least three significant digits, ensuring direct comparability with the relativistic kinetic energy computed previously."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction_5 = "Sub-task 5: Calculate the numerical difference between the relativistic kinetic energy T1_rel and the classical kinetic energy T1_class of the heavier fragment by subtracting the two computed values explicitly. Then, interpret this difference quantitatively and select the correct answer choice from the provided options. Include a verification step to confirm the correctness of the final selection."
    debate_desc5 = {
        'instruction': debate_instruction_5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
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
