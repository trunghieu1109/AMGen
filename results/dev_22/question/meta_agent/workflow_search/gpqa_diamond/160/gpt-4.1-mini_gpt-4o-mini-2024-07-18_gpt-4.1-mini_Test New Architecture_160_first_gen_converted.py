async def forward_160(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given physical parameters and conditions related to the vacuum state, gas molecules, "
        "and electron beam setup, including the definition of λ1 and environmental constants, based on the provided query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Analyze the physical and theoretical relationships between mean free path, pressure, temperature, "
        "and electron scattering effects, clarifying how λ1 and λ2 are defined and how they differ, based on the output from Sub-task 1."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Compute or reason the expected quantitative or conditional relationship between λ2 and λ1 based on electron scattering theory "
        "and vacuum physics, considering that temperature and pressure remain constant, using outputs from Sub-task 2."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Based on the outputs from Sub-tasks 1, 2, and 3, derive the final conclusion about the relationship between λ2 and λ1, "
        "and select the correct choice among the given options."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of subtask 1", "answer of subtask 1",
            "thinking of subtask 2", "answer of subtask 2",
            "thinking of subtask 3", "answer of subtask 3"
        ]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
