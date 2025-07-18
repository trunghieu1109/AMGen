async def forward_180(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Collect and explicitly list detailed, quantitative neutrino energy spectra and flux contributions from each proton-proton chain branch (pp-I, pp-II, pp-III, etc.), focusing on exact neutrino energies and flux magnitudes within the 700-800 keV and 800-900 keV bands. Include authoritative solar neutrino spectral data or references. Perform a numerical verification step to confirm which branches contribute significantly to each energy band, avoiding qualitative assumptions."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc={
            "instruction": cot_instruction1,
            "final_decision_instruction": "Sub-task 1: Determine the detailed neutrino spectral contributions per branch in the specified energy bands.",
            "input": [taskInfo],
            "context_desc": ["user query"],
            "temperature": 0.5
        },
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Analyze how stopping the pp-III branch affects the neutrino flux in the two specified energy bands (700-800 keV and 800-900 keV) using the verified spectral and flux data from Subtask 1. Quantify the expected flux changes in each band, explicitly considering that pp-II (7Be) neutrinos contribute mainly to band 2 and that band 1 is dominated by the continuous tail of pp-III neutrinos. Include a Reflexion step to critically evaluate assumptions and results."
    )
    critic_instruction2 = (
        "Please review and provide the limitations of the analysis on flux changes after stopping the pp-III branch, focusing on assumptions, spectral data reliability, and impact on flux ratio estimation."
    )
    cot_reflect_desc2 = {
        "instruction": cot_reflect_instruction2,
        "critic_instruction": critic_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Calculate or estimate the approximate ratio of neutrino fluxes Flux(band 1) / Flux(band 2) after the pp-III branch has stopped, based on the flux changes analyzed in Subtask 2. Include a verification step comparing the calculated ratio against known solar neutrino flux models or literature to ensure consistency and correctness."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the final approximate ratio Flux(band 1) / Flux(band 2) with verification against solar neutrino models."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
