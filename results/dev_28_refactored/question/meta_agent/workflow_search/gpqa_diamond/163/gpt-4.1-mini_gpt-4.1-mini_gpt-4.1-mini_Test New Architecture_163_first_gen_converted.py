async def forward_163(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and categorize all given information from the query, including orbital periods, radial velocity amplitudes, "
        "and definitions related to the binary systems to establish a clear data foundation."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent extraction and categorization of given data.",
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

    debate_instruction2 = (
        "Sub-task 2: Analyze the relationships between orbital periods, radial velocity amplitudes, and stellar masses "
        "using astrophysical principles (Kepler's laws, mass function, and RV amplitude ratios) to formulate expressions for total masses of system_1 and system_2."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and physically sound expressions for total masses of system_1 and system_2."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the total masses of system_1 and system_2 by applying the derived formulas and given numerical values, "
        "ensuring consistent units and assumptions (e.g., circular orbits, inclination effects)."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": "Sub-task 3: Synthesize and choose the most consistent and accurate computed total masses.",
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
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
        "Sub-task 4: Calculate the ratio of the total mass of system_1 to system_2 and interpret the result to identify the closest matching answer choice from the provided options."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Provide the final answer indicating by what factor system_1 is more massive than system_2, selecting the closest choice."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"],
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
