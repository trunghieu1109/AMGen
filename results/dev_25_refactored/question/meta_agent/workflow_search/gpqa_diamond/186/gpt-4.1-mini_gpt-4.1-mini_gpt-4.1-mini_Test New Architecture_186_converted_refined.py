async def forward_186(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and tabulate the ESPRESSO spectrograph's single 8m VLT UT sensitivity data, "
        "specifically the signal-to-noise ratio (S/N) per binned pixel as a function of apparent V magnitude for a 1-hour exposure in high-resolution mode. "
        "Clarify the meaning and impact of 'binned pixel' on S/N, and gather relevant instrument parameters such as throughput, spectral resolution, detector characteristics, and typical observing conditions from the official ESPRESSO overview and ETC documentation. "
        "Anchor all subsequent S/N calculations to real instrument data."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent and accurate ESPRESSO sensitivity data for further calculations.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Calculate the apparent V magnitudes of all stars listed (Canopus, Polaris, and the four hypothetical stars) "
        "using known data for named stars and the distance modulus formula for hypothetical stars. "
        "Cross-check the consistency of absolute magnitudes and distances, flag borderline or unrealistic cases for further scrutiny."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': "Sub-task 2: Synthesize and choose the most consistent apparent magnitudes for all stars.",
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

    cot_sc_instruction3 = (
        "Sub-task 3: Perform detailed, quantitative S/N estimations per binned pixel for each star during a 1-hour exposure with ESPRESSO on a single 8m VLT UT, "
        "using the apparent magnitudes from subtask_2 and the instrument sensitivity data extracted in subtask_1. "
        "Explicitly incorporate binning factors, instrument throughput, spectral resolution, detector noise, and typical observing conditions. "
        "Produce precise S/N predictions grounded in documented instrument performance."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': "Sub-task 3: Synthesize and choose the most consistent and accurate S/N estimations for all stars.",
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Critically evaluate the physical plausibility and observational constraints for each star, especially borderline cases near the detection threshold. "
        "Assess uncertainties in stellar parameters, instrument efficiency variations, sky conditions, and the impact of binning on S/N. "
        "Integrate these considerations with the quantitative S/N results from subtask_3 to refine detectability predictions, preventing overconfident conclusions and ensuring robust, physically consistent reasoning."
    )
    final_decision_instruction4 = "Sub-task 4: Provide a refined and physically consistent evaluation of detectability for each star."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Apply the detectability criterion (S/N ≥ 10 per binned pixel in 1 hour) strictly to the refined S/N estimates and physical evaluations from subtask_4 to determine which stars are detectable. "
        "Explicitly document the decision process for each star, ensuring transparency and traceability of the detectability classification."
    )
    final_decision_instruction5 = "Sub-task 5: Determine detectability classification for each star based on refined S/N and physical evaluation."
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

    debate_instruction6 = (
        "Sub-task 6: Summarize the total number of detectable stars based on subtask_5's results and map this number to the provided multiple-choice answers (2, 3, 4, or 5). "
        "Clearly justify the chosen answer with reference to the detailed analysis and avoid unsupported assumptions."
    )
    final_decision_instruction6 = "Sub-task 6: Provide final answer mapping the number of detectable stars to the multiple-choice options with justification."
    debate_desc6 = {
        'instruction': debate_instruction6,
        'final_decision_instruction': final_decision_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'context_desc': ["user query", "thinking of subtask 5", "answer of subtask 5"],
        'temperature': 0.5
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])

    return final_answer, logs
