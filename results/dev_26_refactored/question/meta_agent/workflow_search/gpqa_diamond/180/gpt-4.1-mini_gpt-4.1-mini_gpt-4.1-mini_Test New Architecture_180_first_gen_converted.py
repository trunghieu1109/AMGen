async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Identify and summarize the neutrino energy spectra and flux contributions from each branch of the proton-proton chain, "
        "focusing on the pp-III branch and its characteristic neutrino energies, with context from the user query."
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
        "Sub-task 2: Based on the output from Sub-task 1, analyze how stopping the pp-III branch affects the neutrino flux in the two specified energy bands (700-800 keV and 800-900 keV), "
        "considering other branches remain unchanged, with context from the user query and Sub-task 1 outputs."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the effect of stopping the pp-III branch on the neutrino flux ratio."
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

    debate_instruction3 = (
        "Sub-task 3: Using the analyses from Sub-tasks 1 and 2, debate and estimate the approximate ratio of the neutrino fluxes Flux(band 1) / Flux(band 2) after the pp-III branch has stopped, "
        "considering all relevant spectral and flux information."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the final approximate ratio Flux(band 1) / Flux(band 2) after stopping the pp-III branch."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])

    return final_answer, logs
