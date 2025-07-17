async def forward_172(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Calculate the minimum uncertainty in momentum Δp using the Heisenberg uncertainty principle given Δx from the query."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Based on the uncertainty in momentum Δp from Sub-task 1, estimate the uncertainty in energy ΔE of the electron, "
        "considering the electron's kinetic energy and relativistic effects if necessary."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", results1['thinking'], results1['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    reflexion_instruction3 = (
        "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, evaluate and compare the calculated uncertainty in energy ΔE "
        "with the provided multiple-choice options to identify the closest estimate."
    )
    critic_instruction3 = "Please review the validity of the comparison and provide any limitations."
    cot_reflect_desc3 = {
        'instruction': reflexion_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
