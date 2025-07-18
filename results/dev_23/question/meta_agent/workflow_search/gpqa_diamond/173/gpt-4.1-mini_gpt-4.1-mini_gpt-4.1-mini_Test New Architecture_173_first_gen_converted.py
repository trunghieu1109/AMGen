async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and quantify all given physical parameters and relationships: initial mass M, rest-mass energies, "
        "fragment masses (m and 2m), total fragment mass (0.99 M), and mass defect energy (1% of M) from the query. "
        "Provide clear numerical expressions and values where possible."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Using the outputs from Sub-task 1, apply conservation of momentum and energy to express the velocities and momenta "
        "of the two fragments in terms of their known masses and total available kinetic energy. Consider relativistic momentum and energy relations. "
        "Provide detailed step-by-step reasoning and final expressions."
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

    debate_instruction_3 = (
        "Sub-task 3: Calculate the relativistic kinetic energy T1 of the more massive fragment using the relativistic formulas for energy and momentum, "
        "based on the velocities and momenta derived in Sub-task 2. Provide detailed calculations and final numeric value."
    )
    debate_desc3 = {
        'instruction': debate_instruction_3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
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
        "Sub-task 4: Calculate the classical (non-relativistic) kinetic energy T1 of the more massive fragment using classical kinetic energy formulas "
        "and the velocities derived from momentum conservation in Sub-task 2. Provide detailed reasoning and numeric result."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction_5 = (
        "Sub-task 5: Determine the difference between the relativistic and classical kinetic energies of the more massive fragment (T1), "
        "and interpret the result in the context of the problem choices. Provide the final numeric difference and select the closest choice."
    )
    debate_desc5 = {
        'instruction': debate_instruction_5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
