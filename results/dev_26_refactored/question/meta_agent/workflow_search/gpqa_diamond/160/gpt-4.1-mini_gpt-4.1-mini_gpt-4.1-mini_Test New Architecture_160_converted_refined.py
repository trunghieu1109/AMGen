async def forward_160(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given information from the query, including the physical setup, vacuum conditions, initial mean free path λ1, and the observed change to λ2 upon electron beam initiation. "
        "Ensure clear identification of all relevant parameters (pressure, temperature, volume, accelerating voltage) to support subsequent analysis."
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
        "Sub-task 2: Explicitly define and distinguish the physical meanings of mean free path λ1 and λ2: λ1 as the mean free path of gas molecules under ultra-high vacuum conditions, "
        "and λ2 as the effective mean free path of electrons scattering off residual gas molecules. Cite typical scattering cross-section values and explain how these differences affect the mean free paths. "
        "Embed feedback to avoid conflating these concepts and to clarify assumptions such as constant temperature and pressure."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the physical meanings and distinctions of λ1 and λ2."
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
        "Sub-task 3: Perform a quantitative estimation of λ1 and λ2 using standard kinetic theory formulas, typical gas densities at <10^-9 Torr, "
        "and known scattering cross-sections for gas molecule collisions and electron-gas molecule scattering. Use this numerical analysis to determine the correct inequality relationship between λ1 and λ2, "
        "explicitly addressing the flawed assumption that λ2 ≈ 1.22×λ1 or λ2 > λ1. This subtask grounds the conclusion in concrete physics rather than assumptions."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions of quantitative estimation of λ1 and λ2, and verify the physical consistency of the numerical results."
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
        "Sub-task 4: Integrate the qualitative definitions and quantitative estimations to analyze how the electron beam initiation affects the effective mean free path, "
        "and establish the physically consistent theoretical relationship between λ1 and λ2. Critically evaluate the multiple-choice options against this integrated analysis, "
        "ensuring no conceptual errors from previous attempts are repeated. Use Debate to cross-validate reasoning and finalize the correct conclusion about the relationship between λ2 and λ1."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Finalize the correct conclusion about the relationship between λ2 and λ1 based on integrated analysis and debate."
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
