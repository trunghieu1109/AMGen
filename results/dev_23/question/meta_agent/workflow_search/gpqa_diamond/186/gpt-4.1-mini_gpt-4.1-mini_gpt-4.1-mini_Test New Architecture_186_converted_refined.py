async def forward_186(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all relevant input data from the query and the ESPRESSO overview link, "
        "including: (a) apparent magnitudes of known stars Canopus (mV approx -0.7) and Polaris (mV approx 2.0), "
        "(b) absolute magnitudes and distances of hypothetical stars, and "
        "(c) ESPRESSO spectrograph sensitivity parameters or typical performance metrics (e.g., S/N=10 at V~18 in 1 hour exposure on 8m VLT). "
        "Avoid assuming missing data by leveraging all available information and publicly documented instrument performance."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Convert the absolute magnitudes and distances of the hypothetical stars into apparent magnitudes "
        "using the distance modulus formula. Combine these with the known apparent magnitudes of Canopus and Polaris extracted in Subtask 1 "
        "to prepare a complete list of apparent magnitudes for all stars. Cross-validate calculations to avoid errors."
    )
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

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the expected signal-to-noise ratio (S/N) for each star during a 1-hour exposure with ESPRESSO on the 8m VLT, "
        "using the apparent magnitudes from Subtask 2 and the instrument sensitivity parameters from Subtask 1. "
        "Perform numerical estimations or interpolations based on the sensitivity curve, ensuring consistency through cross-validation."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Classify each star as detectable or not based on whether the computed S/N meets or exceeds the threshold of 10 per binned pixel during a 1-hour exposure. "
        "Count the total number of detectable stars. Use Reflexion to ensure consensus and error checking."
    )
    critic_instruction4 = (
        "Please review the classification of stars as detectable or not and provide any limitations or errors found."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Compare the count of detectable stars from Subtask 4 to the provided multiple-choice options and select the correct choice. "
        "Finalize the answer with justification and use Reflexion to confirm correctness and avoid premature conclusions."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
