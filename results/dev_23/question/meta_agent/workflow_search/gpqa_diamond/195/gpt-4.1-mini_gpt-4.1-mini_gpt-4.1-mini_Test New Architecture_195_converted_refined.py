async def forward_195(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Subtask 1: Extract and summarize all given physical information and parameters relevant to the relativistic harmonic oscillator problem, "
        "including mass m, amplitude A, spring constant k, speed of light c, and the four candidate formulas for v_max. "
        "This subtask sets the foundation by clearly identifying all problem data and options to avoid ambiguity in later steps."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Subtask 2: Analyze the relationships between the physical components, including the force law, energy considerations, and relativistic constraints. "
        "Identify key dimensionless parameters influencing v_max and clarify the physical context and assumptions. "
        "Highlight the limitations of classical potential energy assumptions in the relativistic regime and prepare for a rigorous derivation."
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

    debate_instruction3 = (
        "Subtask 3: Derive the relativistic expression for the maximum speed v_max of the oscillator by starting explicitly from the relativistic equation of motion dp/dt = F, "
        "incorporating the velocity-dependent relativistic momentum p = gamma m v and the Hooke's law force F = -kx. "
        "Avoid relying solely on classical potential energy and naive energy conservation. "
        "Critically evaluate and cross-validate assumptions about potential energy and relativistic dynamics to ensure the derivation respects relativistic momentum and force relations. "
        "This subtask addresses the main failure reason of previous attempts by enforcing a rigorous relativistic dynamics approach."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'input': [taskInfo, results1, results2],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Subtask 4: Explicitly restate all four candidate formulas for v_max with their labels (choice1 to choice4). "
        "Symbolically compare the derived expression from Subtask 3 to each candidate formula to identify the exact match. "
        "Confirm that the identified formula respects physical constraints such as v_max < c. "
        "Output the corresponding choice label (e.g., choice2) to avoid any mislabeling errors. "
        "This subtask directly addresses the previous failure of mis-mapping the derived formula to the wrong option by enforcing an explicit matching step."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", results3['thinking'], results3['answer']]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
