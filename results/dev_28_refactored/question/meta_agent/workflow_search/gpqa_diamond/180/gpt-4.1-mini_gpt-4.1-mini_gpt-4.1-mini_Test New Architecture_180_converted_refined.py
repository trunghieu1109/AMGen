async def forward_180(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and categorize all relevant information from the query, including the nature of the neutrino flux, the specific solar nuclear reaction branches (pp-I, pp-II, pp-III), the energy bands of interest (700-800 keV and 800-900 keV), and the assumptions given (e.g., stoppage of pp-III branch, neutrino travel time, ignoring flavor oscillations). Explicitly note the problem context and constraints to avoid misinterpretation."
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

    debate_instruction2 = (
        "Sub-task 2: Collect and integrate authoritative quantitative data on solar neutrino fluxes and energy spectra for each relevant branch (pp-I, pp-II, pp-III). Specifically, integrate the neutrino flux contributions over the two energy bands (700-800 keV and 800-900 keV) for each branch. This subtask addresses the previous failure of assuming qualitative spectral contributions without numerical backing, ensuring that the relative magnitudes of fluxes are clear before further reasoning."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent quantitative flux data for each branch and energy band."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Analyze the impact of stopping the pp-III branch on the neutrino fluxes in each energy band using the quantitative flux integrals from subtask_2. Identify which branches dominate each band and assess how the fluxes change after pp-III cessation. This subtask explicitly avoids the previous error of overestimating pp-III contributions in these bands and ensures a physically accurate assessment of flux changes and their effect on the flux ratio."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 4: Compute the approximate ratio Flux(band 1) / Flux(band 2) after the hypothetical stoppage of the pp-III branch, based on the analysis in subtask_3. Validate the computed ratio against physical expectations and known solar neutrino data. This subtask prevents propagation of earlier errors by grounding the ratio calculation in verified flux contributions."
    )
    cot_agent_desc4 = {
        "instruction": cot_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate the candidate answer choices (0.1, 10, 1, 0.01) against the derived flux ratio from subtask_4. Select the best matching candidate and provide a clear justification referencing the quantitative analysis and spectral dominance. This final evaluation ensures that the answer choice is consistent with the corrected reasoning and data integration, avoiding the previous mistake of selecting an answer based on flawed assumptions."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Select the best answer choice for the flux ratio Flux(band 1) / Flux(band 2) after pp-III stoppage, with justification."
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

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
