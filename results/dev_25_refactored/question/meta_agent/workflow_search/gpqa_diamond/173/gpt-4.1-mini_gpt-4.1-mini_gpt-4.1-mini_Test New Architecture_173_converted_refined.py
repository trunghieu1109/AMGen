async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and transform the given physical parameters into explicit numerical and symbolic forms: "
        "determine the rest masses of the two fragments (m and 2m), total rest mass after fission (0.99 M), and initial conditions for further calculations. "
        "Ensure precise parameter definitions to avoid ambiguity in subsequent calculations."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, solve explicitly for the fragment momentum p by enforcing the relativistic energy-momentum relation: "
        "sqrt(m1^2 c^4 + p^2 c^2) + sqrt(m2^2 c^4 + p^2 c^2) = M c^2 (300 GeV). "
        "Provide the exact or numerical value of p (in GeV/c) and verify the solution's consistency."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the fragment momentum p."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Calculate the relativistic kinetic energy T1_rel of the more massive fragment using the exact momentum p found in subtask 2: "
        "T1_rel = sqrt(m1^2 c^4 + p^2 c^2) - m1 c^2. Include intermediate numerical results (total energy, gamma factor) to ensure transparency."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent relativistic kinetic energy T1_rel value."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="stage_1.subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the classical (non-relativistic) kinetic energy T1_classical of the more massive fragment using the velocity derived from the exact momentum p: "
        "v = p / m1 (non-relativistic approximation), then T1_classical = 1/2 m1 v^2. Use exact momentum and consistent velocity."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent classical kinetic energy T1_classical value."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Stage 2 Sub-task 1: Compute the difference Δ = T1_rel - T1_classical using the results from subtasks 3 and 4. "
        "Analyze the magnitude of Δ carefully to select the closest answer choice from the given options. "
        "Explicitly reference intermediate numerical results to justify the final choice and avoid overestimation errors."
    )
    final_decision_instruction5 = (
        "Stage 2 Sub-task 1: Choose the closest answer choice for the difference Δ between relativistic and classical kinetic energies."
    )
    cot_sc_desc5 = {
        "instruction": cot_sc_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
