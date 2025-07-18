async def forward_172(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and convert all given physical quantities and constants into consistent SI units and prepare them for calculations (electron speed, Δx, electron mass, Planck's constant ħ)."
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
        "Sub-task 2: Apply the Heisenberg uncertainty principle to calculate the minimum uncertainty in momentum Δp from the given Δx, based on outputs from Sub-task 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent calculation for minimum uncertainty in momentum Δp."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
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
        "Sub-task 3: Calculate the uncertainty in kinetic energy ΔE from the uncertainty in momentum Δp, using the relation E = p²/(2m) and considering the electron's known velocity, based on outputs from Sub-tasks 1 and 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent calculation for uncertainty in kinetic energy ΔE."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Compare the calculated minimum uncertainty in energy ΔE with the provided choices and select the closest matching option."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best matching choice for minimum uncertainty in energy ΔE based on calculations."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
