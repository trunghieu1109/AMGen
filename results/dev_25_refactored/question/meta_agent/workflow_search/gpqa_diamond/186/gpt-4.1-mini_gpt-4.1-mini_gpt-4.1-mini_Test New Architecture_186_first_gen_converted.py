async def forward_186(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Convert the given stellar parameters (absolute V magnitude, distance) into apparent V magnitudes for each star, "
        "including Canopus and Polaris using known data, and the hypothetical stars using the distance modulus formula. "
        "Provide detailed Chain-of-Thought reasoning for each star."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the apparent magnitudes from Sub-task 1, estimate the expected signal-to-noise ratio (S/N) per binned pixel "
        "for each star during a 1-hour exposure with the ESPRESSO spectrograph on an 8m VLT telescope. "
        "Use instrument sensitivity data from the provided ESPRESSO overview. Provide multiple reasoning chains and synthesize the most consistent S/N estimates."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent S/N estimates for each star based on multiple reasoning chains."
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
        "Sub-task 3: Integrate and combine the stellar S/N estimates with observational constraints such as sky position, instrument efficiency, and exposure time "
        "to refine detectability predictions for each star. Review limitations and improve the solution accordingly."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of the provided S/N estimates and observational constraints integration, "
        "and refine the detectability predictions for each star."
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
        "Sub-task 4: Evaluate each star against the detectability criterion (S/N ≥ 10 per binned pixel in 1 hour) "
        "and select those stars that meet or exceed this threshold. Provide detailed debate reasoning."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Decide which stars are detectable based on the refined S/N estimates and detectability criterion."
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

    debate_instruction5 = (
        "Sub-task 5: Summarize the total number of detectable stars and map this result to the provided multiple-choice answers (2, 3, 4, or 5). "
        "Provide debate reasoning and final decision."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final answer for the number of detectable stars based on previous evaluation."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
