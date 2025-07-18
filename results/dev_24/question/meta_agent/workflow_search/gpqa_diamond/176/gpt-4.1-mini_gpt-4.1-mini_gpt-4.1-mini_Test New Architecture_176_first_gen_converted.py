async def forward_176(self, taskInfo):
    logs = []

    debate_instruction1 = "Sub-task 1: Extract and classify all given physical parameters and observational data: radius ratio, mass ratio, peak wavelength equality, radial velocities, and black body assumption. Identify which parameters directly affect luminosity and which provide context, based on the user query."
    debate_final_decision1 = "Sub-task 1: Provide a clear classification of parameters affecting luminosity and those providing context."
    debate_desc1 = {
        "instruction": debate_instruction1,
        "final_decision_instruction": debate_final_decision1,
        "input": [taskInfo],
        "context_desc": ["user query"],
        "temperature": 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze the physical relationships between the parameters, especially how radius and temperature (from peak wavelength) relate to luminosity via the Stefan-Boltzmann law. Clarify the role of radial velocity and mass in the luminosity comparison under black body assumptions."
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent and correct explanation for the relationships and roles of parameters affecting luminosity."
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Compute the luminosity ratio of Star_1 to Star_2 using the radius ratio and equal temperature, applying the Stefan-Boltzmann law. Compare the computed factor with the given choices and select the closest approximate value."
    debate_final_decision3 = "Sub-task 3: Provide the computed luminosity ratio and select the closest choice from the given options."
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": debate_final_decision3,
        "input": [taskInfo, results1.get('thinking', ''), results1.get('answer', ''), results2.get('thinking', ''), results2.get('answer', '')],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3.get('thinking', ''), results3.get('answer', ''))
    return final_answer, logs
