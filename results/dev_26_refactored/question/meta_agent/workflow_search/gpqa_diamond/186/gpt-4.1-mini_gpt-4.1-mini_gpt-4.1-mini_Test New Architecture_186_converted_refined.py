async def forward_186(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given star data (including absolute magnitudes, distances, RA/DEC), "
        "instrument parameters, and detectability criteria from the query. Ensure clarity on assumptions such as spectral types, pixel binning, and observational conditions. "
        "This subtask must avoid oversimplified assumptions and prepare comprehensive inputs for further analysis."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Convert absolute magnitudes and distances of stars into apparent magnitudes using the distance modulus formula. "
        "Verify calculations carefully and document assumptions. This subtask depends on the accurate extraction of star data from Subtask 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for apparent magnitude calculations."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Extract detailed ESPRESSO spectrograph sensitivity data from the provided instrument overview link, "
        "including numerical S/N versus apparent magnitude curves, throughput, spectral response, noise characteristics, and pixel binning effects. "
        "This subtask must avoid generic or assumed sensitivity thresholds and produce a concrete, quantitative sensitivity model for use in S/N calculations."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Using the apparent magnitudes from Subtask 2 and the detailed ESPRESSO sensitivity data from Subtask 3, "
        "perform explicit quantitative S/N calculations for each star during a 1-hour exposure on an 8m VLT telescope. "
        "Incorporate assumptions about spectral types, pixel binning, and observational conditions to ensure realistic S/N estimates. "
        "Avoid oversimplified magnitude cutoffs and justify all assumptions clearly."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent S/N calculation results for each star."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate each star's calculated S/N against the detectability threshold (S/N ≥ 10 per binned pixel) "
        "and select those stars that meet or exceed this criterion. This subtask must critically assess the results from Subtask 4 "
        "and ensure no oversimplified assumptions affect the detectability decision."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Select detectable stars based on S/N calculations and justify the selection."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Count the number of detectable stars identified in Subtask 5 and compare this count with the provided answer choices. "
        "Provide a justified final answer selection based on the rigorous S/N calculations and detectability evaluation. "
        "Document reasoning clearly to avoid previous errors of unsupported conclusions."
    )
    final_decision_instruction6 = (
        "Sub-task 6: Provide the final answer choice with justification."
    )
    debate_desc6 = {
        "instruction": debate_instruction6,
        "final_decision_instruction": final_decision_instruction6,
        "input": [taskInfo, results5["thinking"], results5["answer"]],
        "context_desc": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "temperature": 0.5
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6["thinking"], results6["answer"])
    return final_answer, logs
