async def forward_176(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and clarify all given physical parameters and assumptions, including radius ratio, mass ratio, equal peak wavelengths, radial velocities, and black body radiation assumptions, with context from the query."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent clarification of given parameters and assumptions.",
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
        "Sub-task 2: Based on the output from Sub-task 1, apply Wien's displacement law and Stefan-Boltzmann law to relate the given parameters to temperature and luminosity, and clarify the role of radial velocity in the problem context."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": "Sub-task 2: Synthesize and choose the most consistent physical interpretation and calculations.",
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
        "Sub-task 3: Combine the clarified parameters and physical laws from previous subtasks to compute the luminosity ratio of Star_1 to Star_2, considering equal temperatures and radius ratio, and evaluate the impact of radial velocity if any."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": "Sub-task 3: Synthesize and choose the most consistent and accurate luminosity ratio calculation.",
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
        "Sub-task 4: Select the answer choice that best matches the computed luminosity ratio factor from the given options, based on the previous subtasks' outputs."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": "Sub-task 4: Choose the best matching answer choice for the luminosity ratio.",
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
