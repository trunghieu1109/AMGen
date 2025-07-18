async def forward_159(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize the given information, including the geometry of the aperture, the physical setup, and the limit as N approaches infinity."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
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
        "Sub-task 2: Based on the output from Sub-task 1, apply the transformation that the polygonal aperture with equal apothems tends to a circular aperture of radius a as N approaches infinity."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the polygon to circle aperture transformation."
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

    cot_instruction3 = (
        "Sub-task 3: Integrate knowledge of Fraunhofer diffraction for a circular aperture to identify the angular positions of the first two minima in the far-field diffraction pattern using the small-angle approximation."
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Select the correct angular distance between the first two minima from the given choices based on the derived theoretical expression for a circular aperture diffraction pattern."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Choose the best answer for the angular distance between the first two minima in the diffraction pattern of a circular aperture."
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
