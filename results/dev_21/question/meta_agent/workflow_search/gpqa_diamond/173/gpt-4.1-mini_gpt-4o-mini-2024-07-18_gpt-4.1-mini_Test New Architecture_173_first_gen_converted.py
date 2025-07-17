async def forward_173(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and define all relevant physical quantities and parameters from the problem statement, "
        "including initial mass M, fragment rest masses, total rest mass after fission, and the mass ratio between fragments. "
        "Use debate agent collaboration to ensure correctness and completeness."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Apply conservation of momentum and energy principles to relate the velocities and momenta of the two fragments, "
        "and express their kinetic energies in terms of known quantities. Use debate agent collaboration to verify relations."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Calculate the relativistic kinetic energy T1 of the more massive fragment using the relativistic energy-momentum relations "
        "and the parameters established in stage 1. Use self-consistency chain-of-thought to explore possible calculation paths and ensure accuracy."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Calculate the classical (non-relativistic) kinetic energy T1 of the more massive fragment using classical kinetic energy formulas "
        "and the same parameters. Use self-consistency chain-of-thought to verify and refine the calculation."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Determine the difference between the relativistic and classical kinetic energy values of the more massive fragment, "
        "and interpret the result in the context of the problem choices. Use debate agent collaboration to finalize the answer."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
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
