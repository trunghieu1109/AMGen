async def forward_180(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []}, "stage_0.subtask_2": {"thinking": [], "answer": []}}
    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize all relevant information from the query about solar neutrino flux, the hypothetical stoppage of the pp-III branch, the specified neutrino energy bands, and assumptions, ensuring clarity on the problem context and avoiding oversimplification. Input content: taskInfo."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])
        cot_instruction_0_2 = (
            "Sub-task 2: Identify and categorize the neutrino energy bands (700-800 keV and 800-900 keV) and their expected contributions from different pp chain branches, with a focus on the pp-III (8B) neutrino spectrum. This includes gathering or approximating the differential energy spectrum of pp-III neutrinos to enable quantitative analysis, explicitly addressing the previous failure of oversimplified band assignments. Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1 iterations."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Quantitatively estimate the fractional contributions of the pp-III (8B) neutrino flux within each energy band (700-800 keV and 800-900 keV) by integrating or approximating the known 8B neutrino spectrum over these ranges. This subtask addresses the previous missing quantification and ensures the flux contributions are numerically grounded. Input content: taskInfo, all thinking and answers from stage_0.subtask_2 iterations."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for quantitative estimation of pp-III neutrino flux contributions in the two energy bands."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Critically evaluate and verify the assumption that the pp-III neutrino flux is negligible in the 700-800 keV band, using the quantitative results from stage_1.subtask_1. This subtask prevents the previous error of ignoring pp-III contributions in the lower band and ensures the reasoning is consistent with spectral data before proceeding. Input content: taskInfo, thinking and answer from stage_1.subtask_1."
    )
    final_decision_instruction_1_2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for verification of pp-III neutrino flux assumption in the 700-800 keV band."
    )
    cot_sc_desc_1_2 = {
        "instruction": cot_sc_instruction_1_2,
        "final_decision_instruction": final_decision_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc_1_2,
        n_repeat=self.max_sc
    )
    logs.append(log_1_2)
    cot_instruction_2_1 = (
        "Sub-task 1: Analyze the impact of stopping the pp-III branch on the neutrino fluxes in both energy bands, incorporating the quantitative spectral contributions and verified assumptions. Compute the approximate flux ratio Flux(band 1) / Flux(band 2) after the stoppage, ensuring the calculation is based on integrated spectral data and consistent reasoning. Input content: taskInfo, thinking and answer from stage_0.subtask_2 and stage_1.subtask_2."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"]] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)
    cot_agent_instruction_3_1 = (
        "Sub-task 1: Format the final answer as the approximate flux ratio Flux(band 1) / Flux(band 2) and select the correct choice from the given options, based on the computed ratio from stage_2.subtask_1. Input content: taskInfo, thinking and answer from stage_2.subtask_1."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.answer_generate(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_agent_desc_3_1
    )
    logs.append(log_3_1)
    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs