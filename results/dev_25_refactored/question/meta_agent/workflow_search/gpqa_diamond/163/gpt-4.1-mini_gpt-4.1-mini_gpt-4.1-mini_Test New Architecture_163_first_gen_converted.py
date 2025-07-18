async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and transform the given observational data (orbital periods and radial velocity amplitudes) "
        "into parameters suitable for mass calculation, assuming circular orbits and edge-on inclination, "
        "with context from the user query."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Combine the transformed parameters from Sub-task 1 to calculate the mass ratios "
        "of the individual stars in each system using the inverse relation of RV amplitudes and derive the total mass expressions "
        "for system_1 and system_2 using Kepler's third law, with context from previous outputs."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the mass ratio and total mass expressions."
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
        "Sub-task 3: Integrate the mass ratio and total mass expressions from Sub-task 2 to compute the numerical total masses "
        "of system_1 and system_2, and then calculate the factor by which system_1 is more massive than system_2, "
        "with context from previous subtasks."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions for the mass calculation and ratio estimation."
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
        "Sub-task 4: Select the closest approximate factor from the given multiple-choice options (~0.4, ~0.7, ~0.6, ~1.2) "
        "that matches the computed mass ratio of system_1 to system_2, with context from previous subtasks."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Choose the best matching approximate factor for the mass ratio of system_1 to system_2."
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
