async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given physical quantities and parameters from the problem statement, "
        "including initial mass M, rest-mass energies, fragment mass ratio, and rest-mass deficit. "
        "Ensure clarity on assumptions such as ignoring electrons and the exact meaning of 'correct T1'. "
        "This subtask avoids errors from incomplete or ambiguous parameter extraction."
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
        "Sub-task 2: Derive the rest masses of the two fragments from the given mass ratio and total rest-mass deficit. "
        "Formulate conservation of momentum and energy constraints for the two-fragment system, explicitly stating the relativistic energy-momentum relations. "
        "Avoid classical approximations at this stage to prevent propagation of errors."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the derivation of fragment masses and constraints."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Formulate the relativistic and classical kinetic energy expressions for the more massive fragment (T1) in terms of fragment masses and momentum p. "
        "Clearly define the difference ΔT = T1_rel - T1_classical. Emphasize that classical kinetic energy uses (1/2)mv² and relativistic kinetic energy uses E - mc² with E = sqrt(p²c² + m²c⁴). "
        "This subtask sets up the formulas without numerical evaluation."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and finalize the kinetic energy expressions and difference formula."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Solve for the fragment momentum p by enforcing the full relativistic energy sum condition: "
        "E1 + E2 = 300 GeV, where E1 = sqrt(p² + (2m)²) and E2 = sqrt(p² + m²). "
        "This must be done numerically or via sufficiently accurate series expansions (including second-order or higher terms) to avoid classical approximations. "
        "Explicitly avoid using classical momentum or truncated expansions. Provide step-by-step numerical solution with intermediate values and verify physical consistency (e.g., check that velocities are small compared to c, gamma factors are close to 1). "
        "This subtask addresses the core failure in previous attempts."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the final numerical solution for momentum p with verification."
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

    cot_reflect_instruction5 = (
        "Sub-task 5: Compute the relativistic kinetic energy T1_rel and classical kinetic energy T1_classical of the more massive fragment using the momentum p obtained in Subtask 4. "
        "Calculate their difference ΔT with high numerical precision. Include verification steps to confirm that the difference is physically plausible and consistent with expected relativistic corrections (likely small, on the order of MeV or less). "
        "This subtask must explicitly connect back to the formulas from Subtask 3 and numerical results from Subtask 4."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions of kinetic energy calculations and difference ΔT."
    )
    cot_reflect_desc5 = {
        "instruction": cot_reflect_instruction5,
        "critic_instruction": critic_instruction5,
        "input": [taskInfo, results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    cot_instruction6 = (
        "Sub-task 6: Compare the computed difference ΔT between relativistic and classical kinetic energies with the given answer choices. "
        "Select the best matching option and provide a concise justification referencing the numerical results and physical reasoning. "
        "This subtask ensures the final answer is grounded in the corrected calculations and avoids overestimation errors from previous attempts."
    )
    cot_agent_desc6 = {
        "instruction": cot_instruction6,
        "input": [taskInfo, results5["thinking"], results5["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6, log6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6["thinking"], results6["answer"])
    return final_answer, logs
