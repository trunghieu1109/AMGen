async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and transform the given observational data (orbital periods and radial velocity amplitudes) "
        "into parameters suitable for mass calculation, including determining mass ratios from RV amplitudes, "
        "with context from the user query."
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

    debate_instruction2 = (
        "Sub-task 2: Combine the transformed parameters from Sub-task 1 using Kepler's third law and the relation "
        "between radial velocity amplitudes and mass ratios to calculate the total mass of each binary system. "
        "Debate the possible interpretations and calculations to reach a consistent conclusion."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and accurate total mass calculations for system_1 and system_2."
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

    debate_instruction3 = (
        "Sub-task 3: Select and compute the factor by which system_1 is more massive than system_2 by comparing their total masses "
        "from Sub-task 2, and identify the closest matching choice from the given options. Debate to finalize the best answer."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide the final answer for the mass factor comparison and select the closest choice option."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
