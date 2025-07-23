async def forward_179(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and organize all given physical parameters and constraints, including charge values, positions, and geometric constraints, to form a clear intermediate representation of the system, with context from taskInfo."
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

    debate_instruction2 = (
        "Sub-task 2: Determine the minimal energy geometric configuration of the 12 charges on the sphere surface, referencing known solutions such as the Thomson problem or Platonic solids, to approximate the arrangement that minimizes repulsive energy, using outputs from subtask_1."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
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
        "Sub-task 3: Calculate the electrostatic potential energy contributions: (a) between the central charge and each of the 12 charges on the sphere, and (b) between all pairs of the 12 charges on the sphere, using Coulomb's law and the minimal energy configuration from subtask_2, with inputs from subtask_1 and subtask_2."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "input": [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Sum all pairwise potential energies to obtain the total minimum electrostatic potential energy of the system, and convert the result into Joules with three decimal places of precision, using outputs from subtask_3."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent total minimum electrostatic potential energy value for the system."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": final_decision_instruction4,
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
        "Sub-task 5: Format the computed minimum energy value clearly, compare it against the provided multiple-choice options, and summarize the final answer with justification, using outputs from subtask_4."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide the final answer with justification and comparison to choices."
    )
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
