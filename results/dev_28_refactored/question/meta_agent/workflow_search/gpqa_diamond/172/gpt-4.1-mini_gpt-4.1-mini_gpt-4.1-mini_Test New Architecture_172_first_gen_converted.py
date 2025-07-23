async def forward_172(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize all given physical quantities and constants relevant to the problem, "
        "including electron mass, Planck's constant, speed, and uncertainty in position."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent summary of given physical quantities.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Construct the relationships between uncertainty in position (Δx), uncertainty in momentum (Δp), "
        "and uncertainty in energy (ΔE) using the Heisenberg uncertainty principle and kinetic energy formula, "
        "based on the summary from Sub-task 1."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": "Sub-task 2: Synthesize and choose the most consistent relationship between Δx, Δp, and ΔE.",
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Derive the formula to calculate the minimum uncertainty in energy ΔE from the uncertainty in momentum Δp "
        "and given velocity v, incorporating the electron mass and constants, based on the relationships from Sub-task 2."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": "Sub-task 3: Synthesize and choose the most consistent formula for ΔE.",
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Compute the numerical value of the minimum uncertainty in energy ΔE using the derived formula "
        "and given numerical values from previous subtasks."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": "Sub-task 4: Synthesize and choose the most consistent numerical value for ΔE.",
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate the computed minimum uncertainty in energy ΔE against the provided multiple-choice options "
        "and select the best matching candidate."
    )
    final_decision_instruction5 = "Sub-task 5: Select the best matching multiple-choice answer for ΔE based on computation."

    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
