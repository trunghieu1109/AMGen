async def forward_186(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given star data, instrument parameters, and detectability criteria from the query to prepare inputs for further analysis."
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
        "Sub-task 2: Apply necessary transformations to convert absolute magnitudes and distances of stars into apparent magnitudes, using the distance modulus formula, based on the output from Sub-task 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for apparent magnitudes conversion."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
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
        "Sub-task 3: Integrate the apparent magnitudes with ESPRESSO spectrograph sensitivity data to estimate the expected signal-to-noise ratio (S/N) for each star during a 1-hour exposure on an 8m VLT telescope."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions for estimating S/N for each star using ESPRESSO sensitivity data."
    )
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
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
        "Sub-task 4: Evaluate each star's estimated S/N against the detectability threshold (S/N ≥ 10) and select those stars that meet or exceed this criterion."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Determine which stars are detectable based on S/N estimates and threshold."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3['thinking'], results3['answer']],
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
        "Sub-task 5: Count the number of detectable stars and compare with the provided answer choices to determine the correct selection."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final count of detectable stars and select the correct answer choice."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
