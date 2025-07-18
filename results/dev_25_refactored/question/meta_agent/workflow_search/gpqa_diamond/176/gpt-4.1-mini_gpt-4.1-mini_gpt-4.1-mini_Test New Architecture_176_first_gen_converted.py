async def forward_176(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given physical parameters and conditions from the problem statement, "
        "including radius ratio, mass ratio, peak wavelength equality, radial velocities, and assumptions about black body radiation."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Apply relevant physical laws and transformations to interpret the given data, specifically using Wien's displacement law "
        "to confirm equal temperatures and noting the implications of radial velocity on observed wavelengths."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the interpretation of temperature equality and radial velocity effects."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Combine the extracted parameters and physical laws (Stefan-Boltzmann law) to derive the formula for the luminosity ratio of Star_1 to Star_2, "
        "considering equal temperatures and the radius ratio."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions of the luminosity ratio derivation, "
        "ensuring assumptions and formula correctness."
    )
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Calculate the numerical value of the luminosity ratio using the derived formula and given radius ratio, "
        "and compare the result with the provided answer choices to select the closest factor."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the closest luminosity ratio factor from the given choices based on the calculation."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
